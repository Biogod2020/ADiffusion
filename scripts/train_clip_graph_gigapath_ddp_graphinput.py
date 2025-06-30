# train_clip_graph_gigapath_ddp_simplified.py
import os
import argparse
import random
import glob
import datetime
import time
import math
import sys
import warnings
import re
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

# --- DDP Setup/Cleanup (Unchanged) ---
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

# --- MODIFIED: Simplified Dataset Definition ---
# This class is now much simpler because preprocessing has already been done.
# It no longer needs to construct paths or find feature files in real-time.
class PreprocessedGraphDataset(Dataset):
    def __init__(self, preprocessed_graph_dir, rank=0, world_size=1, seed=42, args=None):
        self.preprocessed_graph_dir = preprocessed_graph_dir
        self.items = []
        self.sample_ids_for_this_rank = []

        if not os.path.isdir(preprocessed_graph_dir):
            raise FileNotFoundError(f"Preprocessed graph directory not found: {preprocessed_graph_dir}")

        all_graph_files = sorted(glob.glob(os.path.join(preprocessed_graph_dir, "*_graph.pt")))
        if rank == 0:
            print(f"Found {len(all_graph_files)} preprocessed graph files.")
        if not all_graph_files:
            raise FileNotFoundError("No graph files found.")

        # --- DDP-safe graph subset selection for debugging (This logic remains the same) ---
        if args and args.debug_num_graphs > 0:
            if world_size > 1:
                selected_graphs = []
                if rank == 0:
                    random.seed(args.seed)
                    num_to_select = min(args.debug_num_graphs, len(all_graph_files))
                    random.shuffle(all_graph_files)
                    selected_graphs = all_graph_files[:num_to_select]
                    print(f"--- DEBUG MODE: Selected {len(selected_graphs)} graphs using seed {args.seed}. ---")
                
                object_list_to_broadcast = [selected_graphs] if rank == 0 else [None]
                dist.broadcast_object_list(object_list_to_broadcast, src=0)
                all_graph_files = object_list_to_broadcast[0]
            else:  # Single GPU
                random.seed(args.seed)
                num_to_select = min(args.debug_num_graphs, len(all_graph_files))
                random.shuffle(all_graph_files)
                all_graph_files = all_graph_files[:num_to_select]
                print(f"--- DEBUG MODE (Single GPU): Selected {len(all_graph_files)} graphs using seed {args.seed}. ---")
        
        # --- CORRECTED LOGIC: All processes scan all files ---
        # The DistributedSampler will handle splitting the work later.
        
        # The progress bar is only active on rank 0 to avoid clutter.
        pbar = tqdm(all_graph_files, desc=f"Rank {rank} Scanning All Graph Paths", disable=(rank != 0))

        # Use a set to automatically handle duplicate sample_ids
        unique_ids = set()

        for graph_path in pbar:
            try:
                # This loop now runs on all processes over all relevant files.
                # It's usually fast as it just builds a list of paths and indices.
                graph_data = torch.load(graph_path, map_location='cpu')

                if not hasattr(graph_data, 'y') or not hasattr(graph_data, 'num_nodes'):
                    if rank == 0:
                        print(f"WARNING: Skipping graph {os.path.basename(graph_path)} as it lacks 'y' or 'num_nodes' attribute.")
                    continue
                
                if hasattr(graph_data, 'sample_id') and graph_data.sample_id:
                    unique_ids.add(graph_data.sample_id)

                if graph_data.y.shape[0] != graph_data.num_nodes:
                    if rank == 0:
                        print(f"WARNING: Skipping graph {os.path.basename(graph_path)} due to mismatch: num_nodes={graph_data.num_nodes} but y.shape[0]={graph_data.y.shape[0]}.")
                    continue
                
                # All processes build the full list of items.
                # The sampler will decide which indices this rank actually gets during training.
                for node_idx in range(graph_data.num_nodes):
                    self.items.append((graph_path, node_idx))

            except Exception as e:
                if rank == 0:
                    print(f"Rank {rank} Warn: Error processing graph {os.path.basename(graph_path)}: {e}")

        # This list will now be identical on all ranks, which is what we need for the log printing.
        self.sample_ids_for_this_rank = sorted(list(unique_ids))

        if rank == 0:
            print(f"--- Dataset Initialization Summary ---")
            print(f"Total graphs scanned: {len(all_graph_files)}")
            print(f"Total items (nodes) created across all graphs: {len(self.items)}")
            if len(self.items) == 0:
                print("ERROR: Dataset is empty after scanning. Please check graph files.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # 重试循环仍然是个好主意
        for _ in range(5):
            graph_path, node_idx_in_graph = self.items[idx]
            try:
                # 加载完整的预处理图对象
                graph_data_obj = torch.load(graph_path, map_location='cpu')

                # **关键改动**: 将目标节点的局部索引作为图的一个新属性
                # 我们不再从中提取 gigapath_feature
                graph_data_obj.target_node_idx = torch.tensor(node_idx_in_graph, dtype=torch.long)
                
                if graph_data_obj is None:
                    raise ValueError("Loaded data is None.")
                
                # 只返回一个包含所有信息的 Data 对象
                return graph_data_obj

            except Exception as e:
                # 如果当前索引失败，尝试一个随机的
                idx = random.randint(0, len(self.items) - 1)
        
        raise RuntimeError(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Could not load a valid data item after multiple attempts.")


# --- Collate Function (Unchanged, it's already compatible) ---
def clip_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None and item[1] is not None and item[2] is not None]
    if not batch: return None, None, None,

    gigapath_features = torch.stack([item[0] for item in batch], dim=0)
    graph_list = [item[1] for item in batch]
    node_indices_in_graphs = torch.stack([item[2] for item in batch], dim=0)

    try:
        # Before creating the batch, remove the large 'y' attribute from individual graphs to save memory
        for g in graph_list:
            if hasattr(g, 'y'):
                del g.y
        graph_batch = Batch.from_data_list(graph_list)
    except Exception as e:
        return None, None, None
        
    global_node_indices = []
    try:
        for i in range(len(batch)):
            start_node_idx = graph_batch.ptr[i]
            original_node_idx = node_indices_in_graphs[i].item()
            global_node_indices.append(start_node_idx + original_node_idx)
        global_node_indices_tensor = torch.tensor(global_node_indices, dtype=torch.long)
    except Exception as e:
        return None, None, None

    return gigapath_features, graph_batch, global_node_indices_tensor


# --- Model Definitions (Unchanged) ---
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
    # --- MODIFIED: forward signature and logic ---
    def forward(self, batch):
        # 1. 从批处理后的'Batch'对象中提取 "图像" 和 "文本" 的原始特征
        # "文本" 部分：通过GNN处理整个批处理的图
        all_graph_node_embeddings = self.graph_conditioner(batch)
        # "图像" 部分：直接从 batch.y 中获取预计算的GigaPath特征
        # batch.y 的形状是 [total_nodes, gigapath_feature_dim]
        all_gigapath_features = batch.y
        # 2. 计算目标节点的全局索引
        # batch.target_node_idx 的形状是 [batch_size]
        # batch.ptr 的形状是 [batch_size + 1]
        # global_indices = batch.ptr[i] + local_index[i] for each i in batch
        global_node_indices = batch.ptr[:-1] + batch.target_node_idx
        # 3. 使用全局索引来选择出我们需要的成对特征
        # 选择目标节点的GigaPath特征 (图像模态)
        target_gigapath_features = all_gigapath_features[global_node_indices]
        # 选择目标节点的GNN嵌入 (文本模态)
        st_embeddings = all_graph_node_embeddings[global_node_indices]
        # 4. 将GigaPath特征通过编码器
        image_embeddings = self.gigapath_encoder(target_gigapath_features)
        # 5. 归一化并返回 (同之前)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        st_embeddings = F.normalize(st_embeddings, p=2, dim=-1)
        
        return image_embeddings, st_embeddings, self.logit_scale.exp()


# --- Checkpoint Utilities (Unchanged) ---
def save_clip_checkpoint(epoch, global_step, clip_model_module, optimizer, lr_scheduler, scaler, args):
    if dist.get_rank() == 0:
        checkpoint_name = (f"{args.checkpoint_filename_prefix}_ep{epoch+1}_step{global_step}_"
                           f"bs{args.batch_size_per_gpu}x{dist.get_world_size()}_lr{args.lr}.pt")
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint = {'epoch': epoch, 'global_step': global_step, 'clip_model_state_dict': clip_model_module.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(), 'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                      'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None, 'args': vars(args)}
        torch.save(checkpoint, checkpoint_path)
        print(f"Rank 0: CLIP Checkpoint saved to {checkpoint_path}")

def load_clip_checkpoint(clip_model_module, optimizer, lr_scheduler, scaler, device, args):
    start_epoch, start_step = 0, 0
    checkpoints = []
    if os.path.isdir(args.checkpoint_dir):
        search_pattern = os.path.join(args.checkpoint_dir, f"{args.checkpoint_filename_prefix}_ep*_step*.pt")
        checkpoints = glob.glob(search_pattern)
    latest_checkpoint_path = None
    if checkpoints:
        try:
            latest_checkpoint_path = max(checkpoints, key=lambda p: int(re.search(r'_step(\d+)_', p).group(1)))
        except (AttributeError, ValueError):
            if dist.get_rank() == 0: print(f"Warning: Could not reliably parse step number, using string max.")
            latest_checkpoint_path = max(checkpoints)
    if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
        if dist.get_rank() == 0: print(f"Resuming CLIP training from checkpoint: {latest_checkpoint_path}")
        map_location = {'cuda:0': f'cuda:{device.index}'}
        checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)
        clip_model_module.load_state_dict(checkpoint['clip_model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint and checkpoint['lr_scheduler_state_dict'] is not None:
             lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
             scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        start_step = checkpoint.get('global_step', 0)
        if dist.get_rank() == 0: print(f"Resuming from Epoch {start_epoch}, Global Step {start_step}")
    else:
        if dist.get_rank() == 0: print("No CLIP checkpoint found. Starting from scratch.")
    return start_epoch, start_step

# --- Main Training Function ---
def train_clip_model(rank, world_size, local_rank, args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (rank == 0)
    amp_enabled = (args.mixed_precision in ["bf16", "fp16"])
    amp_dtype = torch.bfloat16 if amp_enabled and args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))
    if is_main_process: print(f"Using AMP: {amp_enabled}, dtype: {amp_dtype}")
    writer = None
    if is_main_process:
        log_run_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_run_dir)
        print(f"TensorBoard logs at: {log_run_dir}")
        with open(os.path.join(log_run_dir, "config_clip_args.txt"), "w") as f:
            import json; json.dump(vars(args), f, indent=2)

    # --- MODIFIED: Use the new simplified dataset ---
    if is_main_process: print("Initializing simplified CLIP dataset...")
    try:
        dataset = PreprocessedGraphDataset(
            args.graph_data_dir, # This now points to the preprocessed directory
            rank, world_size,
            args=args
        )
    except FileNotFoundError as e:
        if is_main_process: print(f"ERROR: Dataset file/dir not found: {e}. Exiting.")
        dist.barrier(); cleanup_ddp(); sys.exit(1)
    
    if world_size > 1:
        dist.barrier()
        if is_main_process: print("All ranks have completed dataset initialization.")

    # ++++++++++++++++ 从这里开始插入新代码 ++++++++++++++++
    # --- 新增：收集并打印所有使用的 Sample ID ---
    if world_size > 1:
        # 在 DDP 环境下, 我们需要从所有进程收集 sample_id 列表
        if is_main_process:
            # Rank 0: 准备一个列表来接收来自所有进程的数据
            gathered_lists = [None] * world_size
            dist.gather_object(dataset.sample_ids_for_this_rank, gathered_lists, dst=0)

            # 将收集到的列表的列表(list of lists)扁平化并去重
            all_sample_ids = set()
            for id_list in gathered_lists:
                if id_list: # 确保列表不为None
                    all_sample_ids.update(id_list)
            
            sorted_ids = sorted(list(all_sample_ids))
        else:
            # 其他 Ranks: 将自己的列表发送给 Rank 0
            dist.gather_object(dataset.sample_ids_for_this_rank, None, dst=0)
            sorted_ids = [] # 其他进程不需要这个列表
    else:
        # 在单 GPU 环境下, 直接使用 dataset 的列表
        sorted_ids = dataset.sample_ids_for_this_rank

    # 只有主进程打印和记录
    if is_main_process:
        log_header = f"--- Training will use {len(sorted_ids)} unique samples ---"
        print(log_header)

        # 为了避免日志过长，可以只打印一部分
        if len(sorted_ids) > 20:
            print(f"Sample IDs (first 20): {sorted_ids[:20]}")
        else:
            print(f"Sample IDs: {sorted_ids}")
        print("-" * (len(log_header)))
        
        # 将完整列表写入 TensorBoard 的 Text 面板，方便永久查阅
        if writer:
            # 使用 Markdown 格式化，在 TensorBoard 中显示效果更好
            tb_text_log = f"### Training Samples ({len(sorted_ids)} total)\n"
            for sample_id in sorted_ids:
                tb_text_log += f"- `{sample_id}`\n"
            writer.add_text("Configuration/Training_Samples", tb_text_log, 0)
    
    # 在所有进程继续之前进行同步，确保 Rank 0 已经完成了打印
    if world_size > 1:
        dist.barrier()
    # --- 结束新增 ---
    # ++++++++++++++++ 在这里结束插入新代码 ++++++++++++++++

    if len(dataset) == 0:
        if is_main_process: print("ERROR: CLIP Dataset is empty. Exiting.")
        dist.barrier(); cleanup_ddp(); sys.exit(1)
        
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = PyGDataLoader(
        dataset, batch_size=args.batch_size_per_gpu, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        # collate_fn=clip_collate_fn # The collate function is still perfectly valid
    )
    if is_main_process: print("CLIP DataLoader initialized.")
    
    # The rest of the training loop is largely unchanged as the data format it receives is the same.
    mlp_dims = [int(d) for d in args.image_encoder_mlp_layers.split(',')]
    gigapath_encoder = GigapathFeatureEncoder(mlp_layers=mlp_dims)
    graph_conditioner = GraphConditioner(
        input_dim=args.conditioner_input_dim, hidden_dim=args.conditioner_hidden_dim,
        output_dim=args.conditioner_output_dim, num_layers=args.conditioner_n_layers,
        heads=args.conditioner_n_heads, attn_dropout=args.conditioner_attn_dropout
    )
    clip_model = GraphGigapathCLIP(gigapath_encoder, graph_conditioner, args.clip_logit_scale_init)
    clip_model.to(device)

    param_groups = [{'params': [p for n, p in clip_model.named_parameters() if p.requires_grad and n != "logit_scale"]},
                    {'params': [clip_model.logit_scale], 'weight_decay': 0.0}]
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6)
    
    num_batches_per_epoch = len(dataloader)
    num_update_steps_per_epoch = math.ceil(num_batches_per_epoch / args.accumulation_steps)
    total_train_steps = args.epochs * num_update_steps_per_epoch
    
    lr_scheduler = None
    if total_train_steps > 0 :
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=args.lr*0.1)
        if is_main_process: print(f"LR Scheduler: CosineAnnealingLR, total steps: {total_train_steps}")
    
    start_epoch, start_step = load_clip_checkpoint(clip_model, optimizer, lr_scheduler, scaler, device, args)
    clip_model = DDP(clip_model, device_ids=[local_rank], find_unused_parameters=False)

    if lr_scheduler and start_step > 0:
        if is_main_process: print(f"Fast-forwarding LR scheduler to step {start_step}...")
        for _ in range(start_step):
            optimizer.step()
            lr_scheduler.step()
        optimizer.zero_grad()
        if is_main_process: print(f"LR Scheduler state advanced. Current LR: {optimizer.param_groups[0]['lr']:.2e}")

    global_step = start_step
    if is_main_process: print(f"\n{'='*20} STARTING SIMPLIFIED CLIP TRAINING {'='*20}")
    # ... Training loop continues as before, it is now independent of the data loading complexity ...
    for epoch in range(start_epoch, args.epochs):
        clip_model.train()
        sampler.set_epoch(epoch)
        epoch_total_loss = 0.0
        epoch_samples_processed = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main_process, leave=True)
        optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None: continue
            
            # gigapath_features, graph_batch, global_node_indices = batch_data
            # gigapath_features = gigapath_features.to(device, non_blocking=True)
            # graph_batch = graph_batch.to(device, non_blocking=True)
            # global_node_indices = global_node_indices.to(device, non_blocking=True)
            
            # current_batch_size = gigapath_features.shape[0]
            
            # 将整个 Batch 对象移动到设备
            batch = batch_data.to(device, non_blocking=True)
            
            # 获取当前批次大小
            current_batch_size = batch.num_graphs
            
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                # image_embeds, text_embeds, logit_scale = clip_model(
                #     gigapath_features, graph_batch, global_node_indices
                # )
                image_embeds, text_embeds, logit_scale = clip_model(batch)
                logits_per_image = logit_scale * image_embeds @ text_embeds.t()
                logits_per_text = logits_per_image.t()

                labels = torch.arange(current_batch_size, device=device, dtype=torch.long)
                
                loss_img = F.cross_entropy(logits_per_image, labels)
                loss_text = F.cross_entropy(logits_per_text, labels)
                loss = (loss_img + loss_text) / 2.0
            
            loss_unscaled = loss.item()
            loss = loss / args.accumulation_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Rank {rank}, Step {global_step}: NaN/Inf loss detected. Skipping batch.")
                optimizer.zero_grad()
                continue

            if scaler.is_enabled(): scaler.scale(loss).backward()
            else: loss.backward()

            epoch_total_loss += loss_unscaled * current_batch_size
            epoch_samples_processed += current_batch_size * world_size

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                if scaler.is_enabled(): scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(clip_model.parameters(), 1.0)

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

        avg_epoch_loss_rank0 = epoch_total_loss / (epoch_samples_processed / world_size) if (epoch_samples_processed > 0) else 0.0
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
        
        dist.barrier()
    
    if is_main_process:
        print(f"\n{'='*20} CLIP TRAINING FINISHED {'='*20}")
        save_clip_checkpoint(args.epochs - 1, global_step, clip_model.module, optimizer, lr_scheduler, scaler, args)
        if writer: writer.close()
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP CLIP Model Training with Preprocessed Graphs")

    # --- MODIFIED: Simplified arguments ---
    parser.add_argument('--graph_data_dir', type=str, required=True, help="Path to the PREPROCESSED graph data directory (files contain data.y).")
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    # --- REMOVED ---
    # The following arguments are no longer needed due to preprocessing.
    # parser.add_argument('--gigapath_feature_dir', type=str, required=True)
    # parser.add_argument('--original_latent_dir_base', type=str, required=True)

    # --- Unchanged Arguments ---
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size_per_gpu', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--mixed_precision', type=str, default='bf16', choices=['bf16', 'fp16', 'no'])
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--conditioner_input_dim', type=int, required=True)
    parser.add_argument('--conditioner_hidden_dim', type=int, required=True)
    parser.add_argument('--conditioner_output_dim', type=int, required=True)
    parser.add_argument('--conditioner_n_layers', type=int, required=True)
    parser.add_argument('--conditioner_n_heads', type=int, required=True)
    parser.add_argument('--conditioner_attn_dropout', type=float, required=True)
    parser.add_argument('--clip_embed_dim', type=int, required=True)
    parser.add_argument('--image_encoder_mlp_layers', type=str, required=True, help="Comma-separated ints for MLP, e.g., '1024,1024,512'")
    parser.add_argument('--clip_logit_scale_init', type=float, required=True)
    parser.add_argument('--save_interval', type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument('--log_interval', type=int, default=50, help="Log basic stats every N optimizer steps")
    parser.add_argument('--checkpoint_filename_prefix', type=str, default="clip_graph_gigapath_preprocessed")
    parser.add_argument('--debug_num_graphs', type=int, default=0, help="Debug: Randomly select N graphs to train on. 0 for all.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # --- NOTE: gigapath_feature_dim and pca_n_comps are no longer direct arguments ---
    # Your model dimensions are now defined by these:
    # - conditioner_input_dim (was likely pca_n_comps)
    # - The first layer of image_encoder_mlp_layers (was likely gigapath_feature_dim)
    # This is a cleaner way to define the model architecture.

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