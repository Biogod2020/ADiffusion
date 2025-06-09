# train_condition_diffusion_ddp_v3.py
import os
import argparse
import random
import glob
import datetime
import time
import math
import sys
import warnings
import io
import re
from pathlib import Path
import PIL.Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GPSConv, GATConv

from diffusers import AutoencoderTiny, UNet2DConditionModel, DDPMScheduler

# --- SCRIPT DESIGN NOTES (V3) ---
# 1. No Custom Collate Fn: This version uses the default PyG DataLoader collation to enhance
#    DDP stability. It combines a batch of Data objects into one large Batch object.
# 2. In-Loop Data Preparation: Data extraction (selecting target latents and conditions)
#    is performed inside the main training loop after the large Batch object is moved to the GPU.
#    This trades some performance for robustness.
# 3. Selective Training: A new argument `--train_sample_id_file` allows restricting training
#    to a specific subset of sample IDs listed in a file.

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- DDP Setup/Cleanup ---
def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
         rank = int(os.environ["RANK"])
         world_size = int(os.environ['WORLD_SIZE'])
         local_rank = int(os.environ['LOCAL_RANK'])
         timeout = datetime.timedelta(minutes=30)
         dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timeout)
         torch.cuda.set_device(local_rank)
         print(f"DDP Setup: Rank {rank}/{world_size}, Device cuda:{local_rank}")
         return rank, world_size, local_rank
    else:
         print("DDP Environment variables not found. Cannot initialize DDP.")
         sys.exit(1)

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# --- Dataset Definition ---
class PreprocessedLatentDataset(Dataset):
    def __init__(self, preprocessed_graph_dir, rank=0, world_size=1, args=None):
        self.preprocessed_graph_dir = preprocessed_graph_dir
        self.items = []

        if not os.path.isdir(preprocessed_graph_dir):
            raise FileNotFoundError(f"Preprocessed graph directory not found: {preprocessed_graph_dir}")

        all_graph_files = sorted(glob.glob(os.path.join(preprocessed_graph_dir, "*_graph.pt")))
        if not all_graph_files:
            raise FileNotFoundError(f"No graph files (*.pt) found in {preprocessed_graph_dir}")

        # --- NEW: Filter graphs based on a list of sample IDs ---
        if args and args.train_sample_id_file:
            if os.path.exists(args.train_sample_id_file):
                if rank == 0:
                    print(f"--- Loading specific sample IDs from: {args.train_sample_id_file} ---")
                with open(args.train_sample_id_file, 'r') as f:
                    allowed_ids = {line.strip() for line in f if line.strip()}
                
                original_count = len(all_graph_files)
                all_graph_files = [
                    p for p in all_graph_files 
                    if os.path.basename(p).replace('_graph.pt', '') in allowed_ids
                ]
                if rank == 0:
                    print(f"Filtered dataset from {original_count} to {len(all_graph_files)} graphs based on the provided ID file.")
                if not all_graph_files:
                    raise ValueError("Filtering resulted in an empty dataset. Check your sample ID file and graph filenames.")
            else:
                raise FileNotFoundError(f"The specified sample ID file does not exist: {args.train_sample_id_file}")
        
        if rank == 0:
            print(f"Scanning {len(all_graph_files)} graph files...")

        pbar = tqdm(all_graph_files, desc=f"Rank {rank} Scanning", disable=(rank != 0))
        for graph_path in pbar:
            try:
                # To speed up scanning, we can try to load only metadata, but for robustness,
                # loading the full object is safer to check all required attributes.
                graph_data = torch.load(graph_path, map_location='cpu')
                if hasattr(graph_data, 'latent') and hasattr(graph_data, 'num_nodes') and graph_data.latent.shape[0] == graph_data.num_nodes:
                    # All processes build the full item list. The Sampler will handle distribution.
                    for node_idx in range(graph_data.num_nodes):
                        self.items.append((graph_path, node_idx))
                else:
                    if rank == 0: print(f"Warning: Skipping {os.path.basename(graph_path)} due to missing/mismatched attributes.")
            except Exception as e:
                if rank == 0: print(f"Warning: Error processing {os.path.basename(graph_path)}: {e}")
        
        if rank == 0:
            print(f"--- Dataset Initialized: {len(self.items)} total valid items (nodes) found. ---")
            if not self.items:
                print("ERROR: Dataset is empty after scanning. Please check graph files and filters.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Retry logic for robustness, e.g., on network file systems
        for _ in range(3):
            graph_path, node_idx_in_graph = self.items[idx]
            try:
                graph_data_obj = torch.load(graph_path, map_location='cpu')
                # Attach the target node index for later use in the training loop
                graph_data_obj.target_node_idx = torch.tensor(node_idx_in_graph, dtype=torch.long)
                return graph_data_obj
            except Exception as e:
                print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: ERROR loading item {idx}. Retrying. Error: {e}")
                idx = random.randint(0, len(self.items) - 1)
        
        raise RuntimeError(f"Could not load a valid data item after multiple attempts.")

# --- Model Definitions ---
class GraphConditioner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, heads, attn_dropout=0.1):
        super().__init__()
        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.output_lin = nn.Linear(hidden_dim, output_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_mpnn = GATConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads, dropout=attn_dropout, add_self_loops=False, concat=True)
            conv = GPSConv(channels=hidden_dim, conv=local_mpnn, heads=heads, dropout=attn_dropout, norm='layer')
            self.convs.append(conv)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, data):
        # This forward pass only uses these attributes, ignoring the large `data.latent`
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.input_lin(x))
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        x = self.norm(x)
        x = self.output_lin(x)
        return x

def get_unet_model(args):
    block_out_channels=tuple(map(int, args.unet_block_out_channels.split(',')))
    down_block_types=tuple(args.unet_down_block_types.split(','))
    up_block_types=tuple(args.unet_up_block_types.split(','))
    return UNet2DConditionModel(
        sample_size=args.unet_sample_size, in_channels=args.unet_in_channels, out_channels=args.unet_out_channels,
        layers_per_block=2, block_out_channels=block_out_channels, down_block_types=down_block_types,
        up_block_types=up_block_types, cross_attention_dim=args.unet_cross_attention_dim,
    )

def get_vae_for_sampling(args):
    if not args.vae_model_path or not os.path.exists(args.vae_model_path):
         print("Rank 0: Warning: VAE path for sampling not found. Using default TAESD.")
         vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32)
    else:
         print(f"Rank 0: Loading VAE for sampling from: {args.vae_model_path}")
         vae = AutoencoderTiny(in_channels=3, out_channels=3, latent_channels=args.unet_in_channels)
         vae.load_state_dict(torch.load(args.vae_model_path, map_location='cpu'))
    if not hasattr(vae.config, 'scaling_factor'): vae.config.scaling_factor = 1.0
    return vae

# --- Sampling Function (Updated for "No Collate" logic) ---
@torch.no_grad()
def sample_and_compare(conditioner, unet_module, vae, scheduler, device, args, global_step, writer):
    conditioner.eval(); unet_module.eval()
    if vae is None:
        print("Rank 0: VAE not available for sampling."); conditioner.train(); unet_module.train(); return
    vae.to(device).eval()

    num_samples = 8
    all_graph_files = glob.glob(os.path.join(args.graph_data_dir, "*_graph.pt"))
    if not all_graph_files:
        print("Rank 0: No graph files found for sampling."); conditioner.train(); unet_module.train(); vae.cpu(); return

    graphs_to_load_paths = random.sample(all_graph_files, min(num_samples, len(all_graph_files)))
    loaded_graphs = [torch.load(p, map_location='cpu') for p in graphs_to_load_paths]
    
    sampling_items = []
    for g in loaded_graphs:
        if len(sampling_items) >= num_samples: break
        node_idx = random.randint(0, g.num_nodes - 1)
        item = g.clone(); item.target_node_idx = torch.tensor(node_idx, dtype=torch.long)
        sampling_items.append(item)

    if not sampling_items:
        print("Rank 0: Could not prepare items for sampling."); conditioner.train(); unet_module.train(); vae.cpu(); return
        
    num_samples = len(sampling_items)
    
    # --- Data preparation mimics the main training loop ---
    sampling_batch = Batch.from_data_list(sampling_items).to(device)
    global_indices = sampling_batch.ptr[:-1] + sampling_batch.target_node_idx
    ground_truth_latents = sampling_batch.latent[global_indices]
    
    all_node_embeddings = conditioner(sampling_batch)
    spot_conditions = all_node_embeddings[global_indices].unsqueeze(1)
    # --- End data preparation ---
    
    latent_shape = ground_truth_latents.shape
    generated_latents = torch.randn(latent_shape, device=device)
    scheduler.set_timesteps(args.sampling_inference_steps)
    
    for t in tqdm(scheduler.timesteps, desc="Sampling", disable=False, leave=False):
        latent_model_input = scheduler.scale_model_input(generated_latents, t)
        noise_pred = unet_module(sample=latent_model_input, timestep=t, encoder_hidden_states=spot_conditions).sample
        generated_latents = scheduler.step(noise_pred, t, generated_latents).prev_sample

    scale_factor = getattr(vae.config, "scaling_factor", 1.0)
    generated_images = vae.decode(generated_latents / scale_factor).sample
    original_images = vae.decode(ground_truth_latents / scale_factor).sample
    generated_images = ((generated_images.clamp(-1, 1) + 1) / 2).cpu().permute(0, 2, 3, 1).numpy()
    original_images = ((original_images.clamp(-1, 1) + 1) / 2).cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(generated_images[i]); axes[i, 0].set_title("Generated"); axes[i, 0].axis('off')
        axes[i, 1].imshow(original_images[i]); axes[i, 1].set_title("Original"); axes[i, 1].axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png'); buf.seek(0)
    if writer: writer.add_image('Step_Comparison_Plot', transforms.ToTensor()(PIL.Image.open(buf)), global_step)
    
    save_dir = os.path.join(args.log_dir, "step_samples")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"comparison_step_{global_step}.png")
    fig.savefig(save_path); plt.close(fig)
    print(f"Rank 0: Saved comparison plot to {save_path}")

    conditioner.train(); unet_module.train(); vae.cpu()


# --- Checkpoint Utilities ---
def save_checkpoint(epoch, global_step, conditioner_module, unet_module, optimizer, lr_scheduler, scaler, args):
    if dist.get_rank() != 0: return
    checkpoint_name = f"{args.checkpoint_filename_prefix}_ep{epoch+1}_step{global_step}.pt"
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save({
        'epoch': epoch, 'global_step': global_step,
        'conditioner_state_dict': conditioner_module.state_dict(),
        'unet_state_dict': unet_module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
        'scaler_state_dict': scaler.state_dict(), 'args': vars(args)
    }, checkpoint_path)
    print(f"Rank 0: Checkpoint saved to {checkpoint_path}")

def load_checkpoint(conditioner, unet, optimizer, lr_scheduler, scaler, device, args):
    start_epoch, start_step = 0, 0
    latest_checkpoint_path = None
    if os.path.isdir(args.checkpoint_dir):
        checkpoints = glob.glob(os.path.join(args.checkpoint_dir, f"{args.checkpoint_filename_prefix}_*.pt"))
        if checkpoints:
            latest_checkpoint_path = max(checkpoints, key=os.path.getctime)

    if not latest_checkpoint_path:
        if dist.get_rank() == 0:
            print("未找到可恢复的checkpoint。")
            if args.pretrained_conditioner_path or args.pretrained_unet_path:
                print("将使用指定的预训练模型权重开始新的训练。")
            else:
                print("从零开始新的训练。")
        # 如果没有checkpoint，就从头加载预训练权重
        if args.pretrained_conditioner_path:
            _load_state_dict_from_checkpoint(conditioner, args.pretrained_conditioner_path, 'clip_model_state_dict', device, model_name_in_clip='graph_conditioner')
        if args.pretrained_unet_path:
            _load_state_dict_from_checkpoint(unet, args.pretrained_unet_path, 'unet_state_dict', device)
        return start_epoch, start_step

    # --- 恢复逻辑 ---
    if dist.get_rank() == 0:
        print(f"正在从checkpoint恢复训练: {os.path.basename(latest_checkpoint_path)}")
    
    map_location = {'cuda:0': f'cuda:{device.index}'}
    checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)
    
    # 始终加载模型权重
    conditioner.load_state_dict(checkpoint['conditioner_state_dict'])
    unet.load_state_dict(checkpoint['unet_state_dict'])
    
    # 获取保存checkpoint时的参数
    args_from_ckpt = checkpoint.get('args', {})
    # 为了兼容旧的checkpoint，如果 'args' 不存在或 'freeze_conditioner' 不存在，给一个默认值
    was_frozen_at_save = vars(args_from_ckpt).get('freeze_conditioner', False) if isinstance(args_from_ckpt, argparse.Namespace) else args_from_ckpt.get('freeze_conditioner', False)

    # 检查当前的 'freeze' 设置是否与保存时一致
    if args.freeze_conditioner == was_frozen_at_save:
        if dist.get_rank() == 0:
            print("`freeze_conditioner` 设置与checkpoint一致。正在恢复优化器状态...")
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        if dist.get_rank() == 0:
            print(f"警告: `freeze_conditioner` 设置已改变 (之前: {was_frozen_at_save}, 现在: {args.freeze_conditioner})。")
            print("将只加载模型权重，优化器状态将从头开始。")

    start_epoch = checkpoint.get('epoch', 0) + 1
    start_step = checkpoint.get('global_step', 0)
    
    if dist.get_rank() == 0:
        print(f"成功恢复模型权重。将从 Epoch {start_epoch}, Step {start_step} 开始。")
        
    return start_epoch, start_step


def _load_state_dict_from_checkpoint(model, checkpoint_path, model_key_in_ckpt, device, model_name_in_clip=None):
    """
    A more robust loading function that correctly handles CLIP-style checkpoints.
    """
    if not os.path.exists(checkpoint_path):
        if dist.get_rank() == 0:
            print(f"Warning: Pretrained checkpoint path not found: {checkpoint_path}")
        return

    if dist.get_rank() == 0:
        print(f"Loading pretrained weights for {model.__class__.__name__} from {checkpoint_path}...")

    map_location = {'cuda:0': f'cuda:{device.index}'}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    state_dict_to_load = None
    if isinstance(checkpoint, dict):
        # --- CORRECTED LOGIC ---
        # 1. Prioritize extracting from a CLIP-style composite checkpoint if requested
        if model_name_in_clip and 'clip_model_state_dict' in checkpoint:
            if dist.get_rank() == 0:
                print(f"Detected CLIP checkpoint. Extracting weights for '{model_name_in_clip}'...")
            clip_state_dict = checkpoint['clip_model_state_dict']
            prefix = model_name_in_clip + '.'
            # Filter and strip prefix
            state_dict_to_load = {k.replace(prefix, ''): v for k, v in clip_state_dict.items() if k.startswith(prefix)}
            if not state_dict_to_load:
                 if dist.get_rank() == 0: print(f"Warning: No keys found with prefix '{prefix}' in 'clip_model_state_dict'.")
        # 2. If not, check for a direct key
        elif model_key_in_ckpt and model_key_in_ckpt in checkpoint:
            if dist.get_rank() == 0:
                print(f"Found weights in checkpoint under key: '{model_key_in_ckpt}'")
            state_dict_to_load = checkpoint[model_key_in_ckpt]
        # 3. Fallback: assume the whole file is the state_dict
        else:
            if dist.get_rank() == 0:
                print("Warning: Neither specific key nor CLIP structure found. Assuming entire file is the state_dict.")
            state_dict_to_load = checkpoint
    else:
        # If the file is not a dictionary, it must be the state_dict itself
        state_dict_to_load = checkpoint

    if not state_dict_to_load:
        if dist.get_rank() == 0:
            print(f"Error: Could not extract a valid state_dict from {checkpoint_path}.")
        return

    # Remove 'module.' prefix (from DDP) if it exists
    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict_to_load.items()}

    try:
        # Use strict=False to report issues without crashing, then check the result
        incompatible_keys = model.load_state_dict(cleaned_state_dict, strict=True)
        if dist.get_rank() == 0:
            print(f"Successfully loaded pretrained weights into {model.__class__.__name__}.")
    except RuntimeError as e:
        if dist.get_rank() == 0:
            print(f"Error: Loading state_dict failed for {model.__class__.__name__}. Model architecture may not match the checkpoint.")
            print(f"Error details: {e}")


# --- Main Training Function ---
def train_diffusion_model(rank, world_size, local_rank, args):
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (rank == 0)

    amp_enabled = (args.mixed_precision in ["bf16", "fp16"])
    amp_dtype = torch.bfloat16 if amp_enabled and args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))

    writer = None
    if is_main_process:
        log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        args.log_dir = log_dir # Update for sampling function to use the correct run-specific folder
        with open(os.path.join(log_dir, "config_args.txt"), "w") as f:
            import json; json.dump(vars(args), f, indent=2)

    dataset = PreprocessedLatentDataset(args.graph_data_dir, rank, world_size, args)
    if len(dataset) == 0:
        if is_main_process: print("FATAL: Dataset is empty. Aborting training."); sys.exit(1)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = PyGDataLoader(
        dataset, batch_size=args.batch_size_per_gpu, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=None # Use PyG's default collate, which creates one large Batch object
    )

    conditioner = GraphConditioner(
        input_dim=args.conditioner_input_dim, hidden_dim=args.conditioner_hidden_dim,
        output_dim=args.conditioner_output_dim, num_layers=args.conditioner_n_layers,
        heads=args.conditioner_n_heads, attn_dropout=args.conditioner_attn_dropout
    )
    unet = get_unet_model(args)
    conditioner.to(device)
    unet.to(device)
    
    # <<< --- REVISED AND CORRECTED LOGIC --- >>>

    # 1. SETUP VAE (for sampling) and DETERMINE PARAMETERS TO OPTIMIZE
    vae = get_vae_for_sampling(args) if is_main_process else None
    
    if args.freeze_conditioner:
        if is_main_process:
            print("--- Freezing GraphConditioner weights. Only the UNet will be trained. ---")
        for param in conditioner.parameters():
            param.requires_grad = False
        params_to_optimize = list(unet.parameters())
    else:
        if is_main_process:
            print("--- Training both Conditioner and UNet. ---")
        params_to_optimize = list(unet.parameters()) + list(conditioner.parameters())

    # 2. CREATE THE OPTIMIZER AND SCHEDULER **BEFORE** LOADING THE CHECKPOINT
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=1e-4)
    total_train_steps = args.epochs * math.ceil(len(dataloader) / args.accumulation_steps)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=1e-7) if total_train_steps > 0 else None

    # 3. NOW, IT IS SAFE TO LOAD THE CHECKPOINT
    # The function can now correctly receive the optimizer and lr_scheduler objects
    # and load their states if a checkpoint exists.
    start_epoch, start_step = load_checkpoint(conditioner, unet, optimizer, lr_scheduler, scaler, device, args)

    # 4. WRAP MODELS WITH DDP
    # conditioner = DDP(conditioner, device_ids=[local_rank])
    unet = DDP(unet, device_ids=[local_rank])
    # 4. WRAP MODELS WITH DDP (CONDITIONALLY)
    if not args.freeze_conditioner:
        # Only wrap the conditioner if it's being trained
        conditioner = DDP(conditioner, device_ids=[local_rank])
    
    # The UNet is always trained, so it's always wrapped
    unet = DDP(unet, device_ids=[local_rank])


    # 5. FAST-FORWARD SCHEDULER IF RESUMING
    if lr_scheduler and start_step > 0:
        if is_main_process: print(f"Fast-forwarding LR scheduler to step {start_step}...")
        for _ in range(start_step): lr_scheduler.step()

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.scheduler_train_timesteps)
    global_step = start_step
    
    if is_main_process:
        print(f"\n{'='*20} STARTING DIFFUSION TRAINING (V3 - No Collate) {'='*20}")
        print(f"Total training steps estimated: {total_train_steps}")

    for epoch in range(start_epoch, args.epochs):
        conditioner.train(); unet.train(); sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main_process)
        optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None: continue
            
            # --- In-Loop Data Preparation (Robust DDP Approach) ---
            batch_data = batch_data.to(device, non_blocking=True)
            global_node_indices = batch_data.ptr[:-1] + batch_data.target_node_idx
            latents = batch_data.latent[global_node_indices]
            
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                all_node_embeddings = conditioner(batch_data)
                spot_conditions = all_node_embeddings[global_node_indices].unsqueeze(1)
            # --- End Data Preparation ---

            current_batch_size = latents.shape[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (current_batch_size,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                noise_pred = unet(sample=noisy_latents, timestep=timesteps, encoder_hidden_states=spot_conditions).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())
            
            loss_unscaled = loss.item()
            loss = loss / args.accumulation_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Rank {rank} Step {global_step}: NaN/Inf loss. Skipping batch."); optimizer.zero_grad(); continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(conditioner.parameters()) + list(unet.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler: lr_scheduler.step()
                global_step += 1

                if is_main_process and global_step % args.log_interval == 0:
                    if writer:
                        writer.add_scalar('Train/Batch_Loss', loss_unscaled, global_step)
                        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                
                if is_main_process and global_step > 0 and global_step % args.sample_interval_steps == 0:
                    print(f"\n--- Generating samples at Step {global_step} ---")
                    sample_and_compare(conditioner.module, unet.module, vae, noise_scheduler, device, args, global_step, writer)
                    conditioner.train(); unet.train()

            if is_main_process:
                progress_bar.set_postfix(loss=f"{loss_unscaled:.4f}", step=global_step, lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        
        if is_main_process and ((epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs):
            save_checkpoint(epoch, global_step, conditioner.module, unet.module, optimizer, lr_scheduler, scaler, args)
        
        dist.barrier()
    
    if is_main_process:
        print(f"\n{'='*20} DIFFUSION TRAINING FINISHED {'='*20}")
        if writer: writer.close()
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Conditioned Diffusion Model Training V3 (No Collate, Selective Training)")

    # Data & Paths
    parser.add_argument('--graph_data_dir', type=str, required=True, help="Path to the PREPROCESSED graph data directory.")
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--vae_model_path', type=str, required=True, help="Path to VAE for decoding.")
    parser.add_argument('--pretrained_unet_path', type=str, default=None, help="Optional path to a pretrained UNet.")

    # <<< --- NEW ARGUMENTS START --- >>>
    parser.add_argument('--pretrained_conditioner_path', type=str, default=None, help="Path to a pretrained GraphConditioner checkpoint to initialize with.")
    parser.add_argument('--freeze_conditioner', action='store_true', help="If set, freezes the weights of the conditioner and only trains the UNet.")
    # <<< --- NEW ARGUMENTS END --- >>>

    parser.add_argument('--train_sample_id_file', type=str, default=None, help="Optional path to a .txt file containing sample IDs to train on, one per line.")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size_per_gpu', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--mixed_precision', type=str, default='bf16', choices=['bf16', 'fp16', 'no'])
    parser.add_argument('--num_workers', type=int, default=8)

    # Model Architecture
    parser.add_argument('--conditioner_input_dim', type=int, required=True)
    parser.add_argument('--conditioner_hidden_dim', type=int, required=True)
    parser.add_argument('--conditioner_output_dim', type=int, required=True)
    parser.add_argument('--conditioner_n_layers', type=int, required=True)
    parser.add_argument('--conditioner_n_heads', type=int, required=True)
    parser.add_argument('--conditioner_attn_dropout', type=float, required=True)
    parser.add_argument('--unet_sample_size', type=int, required=True)
    parser.add_argument('--unet_in_channels', type=int, required=True)
    parser.add_argument('--unet_out_channels', type=int, required=True)
    parser.add_argument('--unet_block_out_channels', type=str, required=True, help="Comma-separated")
    parser.add_argument('--unet_down_block_types', type=str, required=True, help="Comma-separated")
    parser.add_argument('--unet_up_block_types', type=str, required=True, help="Comma-separated")
    parser.add_argument('--unet_cross_attention_dim', type=int, required=True)
    
    # Diffusion & Sampling
    parser.add_argument('--scheduler_train_timesteps', type=int, default=1000)
    parser.add_argument('--sampling_inference_steps', type=int, default=50)

    # Logging & Saving
    parser.add_argument('--checkpoint_filename_prefix', type=str, default="cond_unet_v3")
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--sample_interval_steps', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    rank, world_size, local_rank = setup_ddp()
    
    try:
        train_diffusion_model(rank, world_size, local_rank, args)
    except Exception as e:
        print(f"Rank {rank}: Unhandled exception: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
    finally:
        cleanup_ddp()