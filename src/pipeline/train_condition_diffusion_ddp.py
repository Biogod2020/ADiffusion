# train_clip_graph_gigapath_ddp.py
import os
import argparse
import random
import glob
import datetime
import time
import math
import sys
import warnings
import re # 正则表达式库，用于解析文件名
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GPSConv, GATConv

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- DDP Setup/Cleanup (same as your diffusion script) ---
def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
         rank = int(os.environ["RANK"])
         world_size = int(os.environ['WORLD_SIZE'])
         local_rank = int(os.environ['LOCAL_RANK'])
         timeout = datetime.timedelta(minutes=30)
         dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timeout)
         torch.cuda.set_device(local_rank)
    else:
         print("DDP Environment variables not found.")
         sys.exit(1)
    print(f"DDP Setup: Rank {rank}/{world_size}, Local Rank {local_rank}, Device cuda:{local_rank}")
    return rank, world_size, local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP Cleanup Completed.")

# --- Dataset Definition ---
class GraphGigapathFeatureDataset(Dataset):
    def __init__(self, graph_data_dir, gigapath_feature_dir, rank=0, world_size=1):
        self.graph_data_dir = graph_data_dir
        self.gigapath_feature_dir = gigapath_feature_dir
        self.items = []  # Stores (graph_path, node_idx_in_graph, constructed_gigapath_feature_path)

        if not os.path.isdir(graph_data_dir):
            raise FileNotFoundError(f"Graph data directory not found: {graph_data_dir}")
        if not os.path.isdir(gigapath_feature_dir):
            raise FileNotFoundError(f"GigaPath feature directory not found: {gigapath_feature_dir}")

        all_graph_files = sorted(glob.glob(os.path.join(graph_data_dir, "*_graph.pt")))
        if rank == 0: print(f"Found {len(all_graph_files)} graph files.")
        if not all_graph_files: raise FileNotFoundError("No graph files found.")

        # DDP file splitting logic for scanning
        num_files_per_rank = math.ceil(len(all_graph_files) / world_size)
        start_idx = rank * num_files_per_rank
        end_idx = min(start_idx + num_files_per_rank, len(all_graph_files))
        graphs_for_this_rank = all_graph_files[start_idx:end_idx]

        # 正则表达式来从文件名中提取 SAMPLE_ID, X, Y
        # 假设文件名格式是 SAMPLE_ID_X_Y.pt (例如 MEND123_20022_24778.pt)
        # 这个正则表达式需要根据您实际文件名中的分隔符和数字位数进行调整
        # \w+ 匹配 SAMPLE_ID (字母、数字、下划线)
        # \d+ 匹配 X 和 Y 坐标 (数字)
        filename_pattern = re.compile(r"(\w+)_(\d+)_(\d+)\.pt")

        pbar = tqdm(graphs_for_this_rank, desc=f"Rank {rank} Scanning Graphs", disable=(rank != 0))
        processed_graphs_count = 0
        found_items_count = 0
        skipped_due_missing_attr = 0
        skipped_due_no_latent_path_info = 0
        skipped_due_filename_parse_error = 0
        skipped_due_feature_not_exist = 0


        for graph_path in pbar:
            try:
                graph_data = torch.load(graph_path, map_location='cpu')
                processed_graphs_count += 1

                if not hasattr(graph_data, 'latent_paths') or not hasattr(graph_data, 'num_nodes') or not hasattr(graph_data, 'coords'):
                    skipped_due_missing_attr +=1
                    continue

                for node_idx in range(graph_data.num_nodes):
                    # original_path_info 可能包含不正确的路径，但文件名可能是正确的
                    original_path_info = graph_data.latent_paths[node_idx]
                    node_coords = graph_data.coords[node_idx] # 获取当前节点的坐标 (假设是 [x, y] 格式)

                    if original_path_info is None and node_coords is None:
                        skipped_due_no_latent_path_info +=1
                        continue

                    # 优先尝试从 original_path_info 的文件名解析
                    sample_id_from_path, x_coord_from_path, y_coord_from_path = None, None, None
                    if original_path_info:
                        filename_from_path = os.path.basename(original_path_info)
                        match_from_path = filename_pattern.match(filename_from_path)
                        if match_from_path:
                            sample_id_from_path = match_from_path.group(1)
                            x_coord_from_path = match_from_path.group(2)
                            y_coord_from_path = match_from_path.group(3)

                    # 使用 graph_data.sample_id (如果存在) 和 graph_data.coords 作为更可靠的来源
                    # 确保 graph_data 对象有 'sample_id' 属性
                    current_sample_id = getattr(graph_data, 'sample_id', None)
                    if current_sample_id is None and sample_id_from_path:
                        current_sample_id = sample_id_from_path # 回退到从路径解析的ID
                    elif current_sample_id is None:
                        # 如果两者都没有，则无法确定sample_id
                        skipped_due_filename_parse_error +=1
                        continue


                    # 获取坐标，优先使用 graph_data.coords
                    # 假设 graph_data.coords 存储的是 [col, row] 即 [x, y]
                    # 并且需要是整数
                    if node_coords is not None and len(node_coords) >= 2:
                        # 假设 GigaPath 特征文件名中的坐标是整数
                        x_coord_str = str(int(round(node_coords[0].item()))) # X-coordinate (col)
                        y_coord_str = str(int(round(node_coords[1].item()))) # Y-coordinate (row)
                    elif x_coord_from_path and y_coord_from_path: # 回退到从路径解析的坐标
                        x_coord_str = x_coord_from_path
                        y_coord_str = y_coord_from_path
                    else:
                        skipped_due_filename_parse_error +=1
                        continue # 没有可用的坐标信息


                    # 构建 GigaPath 特征文件名
                    # 规则: {GIGAPATH_BASE_PATH}/{SAMPLE_ID}/{SAMPLE_ID}_{X}_{Y}.pt
                    # 注意: 这里的 {SAMPLE_ID} 子目录是根据您的描述添加的。如果您的GigaPath特征直接在
                    # GIGAPATH_BASE_PATH 下，没有按 SAMPLE_ID 分子目录，则移除 os.path.join 中的 current_sample_id
                    expected_feature_filename = f"{current_sample_id}_{x_coord_str}_{y_coord_str}.pt"
                    current_sample_id_dir = f"{current_sample_id}_tiles" # 用于构建路径
                    
                    # 假设 GigaPath 特征的目录结构是 GIGAPATH_FEATURE_DIR / SAMPLE_ID / features.pt
                    # 如果您的 GigaPath 特征文件直接在 GIGAPATH_FEATURE_DIR 下，没有 SAMPLE_ID 子目录，
                    # 则改为： constructed_gp_feature_path = os.path.join(self.gigapath_feature_dir, expected_feature_filename)
                    constructed_gp_feature_path = os.path.join(self.gigapath_feature_dir, current_sample_id_dir, expected_feature_filename)
                    
                    # 另一种可能的结构：GIGAPATH_FEATURE_DIR / SAMPLE_ID_tiles / features.pt
                    # constructed_gp_feature_path = os.path.join(self.gigapath_feature_dir, f"{current_sample_id}_tiles", expected_feature_filename)

                    if os.path.exists(constructed_gp_feature_path):
                        self.items.append((graph_path, node_idx, constructed_gp_feature_path))
                        found_items_count +=1
                    else:
                        skipped_due_feature_not_exist +=1
                        # 可选的调试信息：
                        # if rank == 0 and skipped_due_feature_not_exist < 10: # 只打印少量错误
                        #     print(f"DEBUG: Feature not found at reconstructed path: {constructed_gp_feature_path}")
                        #     print(f"  Based on sample_id='{current_sample_id}', x='{x_coord_str}', y='{y_coord_str}'")
                        #     print(f"  Original path info was: {original_path_info}")
                        #     if node_coords is not None: print(f"  Node coords were: {node_coords.tolist()}")


            except Exception as e:
                if rank == 0: print(f"Rank {rank} Warn: Error processing graph {os.path.basename(graph_path)}: {e}")
        
        if rank == 0:
            print(f"--- Rank {rank} Dataset Scan Summary ---")
            print(f"Processed graphs: {processed_graphs_count}")
            print(f"Total items (node, feature pairs) found: {found_items_count}")
            print(f"Graphs skipped (missing attributes): {skipped_due_missing_attr}")
            print(f"Nodes skipped (no latent/coord info): {skipped_due_no_latent_path_info}")
            print(f"Nodes skipped (filename/coord parse error): {skipped_due_filename_parse_error}")
            print(f"Nodes skipped (constructed feature path not exist): {skipped_due_feature_not_exist}")
            if found_items_count == 0:
                print("ERROR: Rank 0 found NO valid items. Please check GigaPath feature paths and naming conventions carefully.")
                print(f"  Example expected GigaPath feature structure: {self.gigapath_feature_dir}/[SAMPLE_ID]/[SAMPLE_ID]_[X]_[Y].pt")


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        graph_path, node_idx_in_graph, constructed_gigapath_feature_path = self.items[idx]
        try:
            graph_data_obj = torch.load(graph_path, map_location='cpu')
            gigapath_feature = torch.load(constructed_gigapath_feature_path, map_location='cpu')
            return gigapath_feature.float(), graph_data_obj, torch.tensor(node_idx_in_graph, dtype=torch.long)
        except Exception as e:
            # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Error loading item {idx} ({os.path.basename(constructed_gigapath_feature_path)}): {e}")
            return None, None, None

# --- Collate Function (Similar to diffusion script's) ---
def clip_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None and item[1] is not None and item[2] is not None]
    if not batch: return None, None, None

    gigapath_features = torch.stack([item[0] for item in batch], dim=0)
    graph_list = [item[1] for item in batch]
    node_indices_in_graphs = torch.stack([item[2] for item in batch], dim=0)

    try:
        graph_batch = Batch.from_data_list(graph_list)
    except Exception as e:
        # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Error in Batch.from_data_list: {e}")
        return None, None, None
        
    global_node_indices = []
    try:
        for i in range(len(batch)):
            start_node_idx = graph_batch.ptr[i]
            original_node_idx = node_indices_in_graphs[i].item()
            global_node_indices.append(start_node_idx + original_node_idx)
        global_node_indices_tensor = torch.tensor(global_node_indices, dtype=torch.long)
    except Exception as e:
        # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Error calculating global_node_indices: {e}")
        return None, None, None

    return gigapath_features, graph_batch, global_node_indices_tensor


# --- Model Definitions (from conceptual cell 3) ---
class GigapathFeatureEncoder(nn.Module):
    def __init__(self, mlp_layers):
        super().__init__()
        layers = []
        for i in range(len(mlp_layers) - 1):
            layers.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))
            if i < len(mlp_layers) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.projection = nn.Sequential(*layers)
    def forward(self, x): return self.projection(x)

class GraphConditioner(nn.Module): # Copied from your diffusion script
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, heads, attn_dropout=0.1):
        super().__init__()
        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.output_lin = nn.Linear(hidden_dim, output_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_mpnn = GATConv(
                in_channels=hidden_dim, out_channels=hidden_dim // heads,
                heads=heads, dropout=attn_dropout, add_self_loops=False, concat=True
            )
            conv = GPSConv(
                channels=hidden_dim, conv=local_mpnn, heads=heads,
                dropout=attn_dropout, norm='layer'
             )
            self.convs.append(conv)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.input_lin(x))
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        x = self.norm(x)
        x = self.output_lin(x)
        return x

class GraphGigapathCLIP(nn.Module):
    def __init__(self, gigapath_encoder, graph_conditioner, logit_scale_init):
        super().__init__()
        self.gigapath_encoder = gigapath_encoder
        self.graph_conditioner = graph_conditioner
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
    def forward(self, gigapath_features, graph_batch, graph_node_indices):
        image_embeddings = self.gigapath_encoder(gigapath_features)
        all_graph_node_embeddings = self.graph_conditioner(graph_batch)
        text_embeddings = all_graph_node_embeddings[graph_node_indices]
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        return image_embeddings, text_embeddings, self.logit_scale.exp()

# --- Checkpoint Utilities (Adapted for CLIP) ---
def save_clip_checkpoint(epoch, global_step, clip_model_module, optimizer, lr_scheduler, scaler, args):
    if dist.get_rank() == 0:
        checkpoint_name = (
            f"{args.checkpoint_filename_prefix}_ep{epoch+1}_step{global_step}_"
            f"bs{args.batch_size_per_gpu}x{dist.get_world_size()}_lr{args.lr}.pt"
        )
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'clip_model_state_dict': clip_model_module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
            'args': vars(args)
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Rank 0: CLIP Checkpoint saved to {checkpoint_path}")

def load_clip_checkpoint(clip_model_module, optimizer, lr_scheduler, scaler, device, args):
    start_epoch, start_step = 0, 0
    # Find latest checkpoint by global_step
    checkpoints = []
    if os.path.isdir(args.checkpoint_dir):
        search_pattern = os.path.join(args.checkpoint_dir, f"{args.checkpoint_filename_prefix}_ep*_step*.pt")
        checkpoints = glob.glob(search_pattern)
    
    latest_checkpoint_path = None
    if checkpoints:
        latest_checkpoint_path = max(checkpoints, key=lambda p: int(p.split('_step')[-1].split('_')[0]))

    if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
        if dist.get_rank() == 0: print(f"Resuming CLIP training from checkpoint: {latest_checkpoint_path}")
        map_location = {'cuda:0': f'cuda:{device.index}'}
        checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)
        
        clip_model_module.load_state_dict(checkpoint['clip_model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
             scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        start_step = checkpoint.get('global_step', 0)
        if dist.get_rank() == 0: print(f"Resuming from Epoch {start_epoch}, Step {start_step + 1}")
    else:
        if dist.get_rank() == 0: print("No CLIP checkpoint found. Starting from scratch.")
    return start_epoch, start_step


# --- Main Training Function ---
def train_clip_model(rank, world_size, local_rank, args):
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (rank == 0)

    # Mixed Precision
    amp_enabled = (args.mixed_precision in ["bf16", "fp16"])
    amp_dtype = torch.float32
    if amp_enabled:
        if args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported(): amp_dtype = torch.bfloat16
        else: amp_dtype = torch.float16 # Fallback to fp16 if bf16 not supported or fp16 chosen
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))
    if is_main_process: print(f"Using AMP: {amp_enabled}, dtype: {amp_dtype}")

    # TensorBoard
    writer = None
    if is_main_process:
        log_run_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_run_dir)
        print(f"TensorBoard logs at: {log_run_dir}")
        with open(os.path.join(log_run_dir, "config_clip_args.txt"), "w") as f:
            import json; json.dump(vars(args), f, indent=2)

    # Dataset and DataLoader
    if is_main_process: print("Initializing CLIP dataset...")
    try:
        dataset = GraphGigapathFeatureDataset(
            args.graph_data_dir, args.gigapath_feature_dir, rank, world_size
        )
    except FileNotFoundError as e:
        if is_main_process: print(f"ERROR: Dataset file/dir not found: {e}. Exiting.")
        dist.barrier(); cleanup_ddp(); sys.exit(1)
    if len(dataset) == 0:
        if is_main_process: print("ERROR: CLIP Dataset is empty. Exiting.")
        dist.barrier(); cleanup_ddp(); sys.exit(1)
        
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = PyGDataLoader(
        dataset, batch_size=args.batch_size_per_gpu, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=clip_collate_fn
    )
    if is_main_process: print("CLIP DataLoader initialized.")

    # Models
    mlp_dims = [int(d) for d in args.image_encoder_mlp_layers.split(',')]
    gigapath_encoder = GigapathFeatureEncoder(mlp_layers=mlp_dims)
    graph_conditioner = GraphConditioner(
        input_dim=args.conditioner_input_dim, hidden_dim=args.conditioner_hidden_dim,
        output_dim=args.conditioner_output_dim, num_layers=args.conditioner_n_layers,
        heads=args.conditioner_n_heads, attn_dropout=args.conditioner_attn_dropout
    )
    clip_model = GraphGigapathCLIP(gigapath_encoder, graph_conditioner, args.clip_logit_scale_init)
    clip_model.to(device)

    # Optimizer and Scheduler
    # Exclude logit_scale from weight decay if desired (common practice)
    param_groups = [
        {'params': [p for n, p in clip_model.named_parameters() if p.requires_grad and n != "logit_scale"]},
        {'params': [clip_model.logit_scale], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6)
    
    num_batches_per_epoch = len(dataloader)
    num_update_steps_per_epoch = math.ceil(num_batches_per_epoch / args.accumulation_steps)
    total_train_steps = args.epochs * num_update_steps_per_epoch
    
    lr_scheduler = None
    if total_train_steps > 0 :
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=args.lr*0.1) # common to anneal to 10% of lr
        if is_main_process: print(f"LR Scheduler: CosineAnnealingLR, total steps: {total_train_steps}")
    
    # Load Checkpoint
    start_epoch, start_step = load_clip_checkpoint(clip_model, optimizer, lr_scheduler, scaler, device, args)

    # DDP Wrap
    clip_model = DDP(clip_model, device_ids=[local_rank], find_unused_parameters=False) # Set find_unused_parameters carefully

    # Resume LR scheduler state
    if lr_scheduler and start_step > 0:
        if is_main_process: print(f"Fast-forwarding LR scheduler to step {start_step}...")
        # Simulate optimizer steps to advance scheduler
        for _ in range(start_step):
            # AdamW does not require gradients to be present for step, but some optimizers might
            # For safety, could temporarily set dummy grads as in diffusion script if scheduler needs it
            optimizer.step() # Call step (ok for AdamW even without grads if scheduler depends on optimizer step count)
            lr_scheduler.step()
        optimizer.zero_grad()
        if is_main_process: print(f"LR Scheduler state advanced. Current LR: {optimizer.param_groups[0]['lr']:.2e}")


    # Training Loop
    global_step = start_step
    if is_main_process: print(f"\n{'='*20} STARTING CLIP TRAINING {'='*20}")

    for epoch in range(start_epoch, args.epochs):
        clip_model.train()
        sampler.set_epoch(epoch)
        epoch_total_loss = 0.0
        epoch_samples_processed = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main_process, leave=True)
        optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None or batch_data[0] is None: continue
            
            gigapath_features, graph_batch, global_node_indices = batch_data
            gigapath_features = gigapath_features.to(device, non_blocking=True)
            graph_batch = graph_batch.to(device, non_blocking=True)
            global_node_indices = global_node_indices.to(device, non_blocking=True)
            
            current_batch_size = gigapath_features.shape[0]
            
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                image_embeds, text_embeds, logit_scale = clip_model(
                    gigapath_features, graph_batch, global_node_indices
                )
                # Contrastive loss
                logits_per_image = logit_scale * image_embeds @ text_embeds.t()
                logits_per_text = logits_per_image.t()

                labels = torch.arange(current_batch_size, device=device, dtype=torch.long)
                
                loss_img = F.cross_entropy(logits_per_image, labels)
                loss_text = F.cross_entropy(logits_per_text, labels)
                loss = (loss_img + loss_text) / 2.0
            
            loss_unscaled = loss.item() # For logging
            loss = loss / args.accumulation_steps # Scale for accumulation
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Rank {rank}, Step {global_step}: NaN/Inf loss detected: {loss_unscaled}. Skipping batch.")
                # Potentially skip optimizer step or exit
                optimizer.zero_grad() # Clear any accumulated grads from this problematic batch
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_total_loss += loss_unscaled * current_batch_size
            epoch_samples_processed += current_batch_size * world_size # All gpus contribute

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(clip_model.parameters(), 1.0) # Clip all model params

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler: lr_scheduler.step()
                global_step += 1

                if is_main_process and global_step % args.log_interval == 0:
                    writer.add_scalar('CLIP_Train/Batch_Loss_Rank0', loss_unscaled, global_step)
                    writer.add_scalar('CLIP_Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('CLIP_Train/Logit_Scale', logit_scale.item(), global_step)
                    writer.add_scalar('CLIP_Train/Grad_Norm', grad_norm.item(), global_step)
                    if scaler.is_enabled(): writer.add_scalar('CLIP_Train/GradScaler_Scale', scaler.get_scale(), global_step)
            
            if is_main_process:
                progress_bar.set_postfix(loss=f"{loss_unscaled:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}", step=global_step)

        # End of Epoch
        avg_epoch_loss_rank0 = epoch_total_loss / (epoch_samples_processed / world_size) if (epoch_samples_processed > 0) else 0.0
        
        # Gather total loss and samples for accurate average across all ranks
        loss_data_tensor = torch.tensor([epoch_total_loss, epoch_samples_processed], device=device, dtype=torch.float64)
        dist.all_reduce(loss_data_tensor, op=dist.ReduceOp.SUM)
        total_loss_all_ranks = loss_data_tensor[0].item()
        total_samples_all_ranks = loss_data_tensor[1].item()
        avg_epoch_loss_all_ranks = total_loss_all_ranks / total_samples_all_ranks if total_samples_all_ranks > 0 else 0.0


        if is_main_process:
            print(f"\nEpoch {epoch+1} summary: Avg Loss (All Ranks): {avg_epoch_loss_all_ranks:.6f}")
            if writer: writer.add_scalar('CLIP_Train/Epoch_Avg_Loss_All', avg_epoch_loss_all_ranks, epoch + 1)
            if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                save_clip_checkpoint(epoch, global_step, clip_model.module, optimizer, lr_scheduler, scaler, args)
        
        dist.barrier() # Sync before next epoch

    # End of Training
    if is_main_process:
        print(f"\n{'='*20} CLIP TRAINING FINISHED {'='*20}")
        save_clip_checkpoint(args.epochs - 1, global_step, clip_model.module, optimizer, lr_scheduler, scaler, args)
        if writer: writer.close()
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP CLIP Model Training for GigaPath and Graphs")
    # Add arguments based on TrainingConfigCLIP.get_script_args()
    parser.add_argument('--gigapath_feature_dir', type=str, required=True)
    parser.add_argument('--graph_data_dir', type=str, required=True)
    parser.add_argument('--original_latent_dir_base', type=str, required=True, help="Base path to original VAE latents, used for reconstructing GigaPath feature paths if graph.latent_paths are absolute to VAEs.")
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size_per_gpu', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--mixed_precision', type=str, default='bf16', choices=['bf16', 'fp16', 'no'])
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gigapath_feature_dim', type=int, required=True)
    parser.add_argument('--pca_n_comps', type=int, required=True) # For GraphConditioner input_dim consistency
    parser.add_argument('--conditioner_input_dim', type=int, required=True)
    parser.add_argument('--conditioner_hidden_dim', type=int, required=True)
    parser.add_argument('--conditioner_output_dim', type=int, required=True) # This will be CLIP_EMBED_DIM
    parser.add_argument('--conditioner_n_layers', type=int, required=True)
    parser.add_argument('--conditioner_n_heads', type=int, required=True)
    parser.add_argument('--conditioner_attn_dropout', type=float, required=True)
    parser.add_argument('--clip_embed_dim', type=int, required=True)
    parser.add_argument('--image_encoder_mlp_layers', type=str, required=True, help="Comma-separated ints for MLP, e.g., '1024,1024,512'")
    parser.add_argument('--clip_logit_scale_init', type=float, required=True)
    parser.add_argument('--save_interval', type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument('--log_interval', type=int, default=50, help="Log basic stats every N optimizer steps")
    parser.add_argument('--checkpoint_filename_prefix', type=str, default="clip_graph_gigapath")
    
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    try:
        train_clip_model(rank, world_size, local_rank, args)
    except KeyboardInterrupt:
        if rank == 0: print("Training interrupted by user.")
    except Exception as e:
        if rank == 0:
            print(f"Unhandled exception during training: {e}")
            import traceback
            traceback.print_exc()
    finally:
        cleanup_ddp()