# train_condition_diffusion_ddp.py
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
from pathlib import Path
import PIL.Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DistributedSampler # Keep standard DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm # Use standard tqdm in script

# PyTorch Geometric imports
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader # Use PyG's DataLoader
from torch_geometric.nn import GPSConv, GATConv # Import model components

# Diffusers imports
from diffusers import AutoencoderTiny, UNet2DConditionModel, DDPMScheduler

# Filter warnings during training
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- DDP Setup/Cleanup ---
def setup_ddp():
    """Initializes the distributed process group."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
         rank = int(os.environ["RANK"])
         world_size = int(os.environ['WORLD_SIZE'])
         local_rank = int(os.environ['LOCAL_RANK'])
         print(f"DDP ENV VARS: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
         # Use a timeout, e.g., 30 minutes, useful for large datasets/models
         timeout = datetime.timedelta(minutes=30)
         dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timeout)
         torch.cuda.set_device(local_rank)
    else:
         print("DDP Environment variables (RANK, WORLD_SIZE, LOCAL_RANK) not found. Cannot initialize DDP.")
         sys.exit(1)
    print(f"DDP Setup: Rank {rank}/{world_size}, Local Rank {local_rank}, Device cuda:{local_rank}")
    return rank, world_size, local_rank

def cleanup_ddp():
    """Destroys the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP Cleanup Completed.")

# Logger Classh
# Helper class to write to multiple streams (e.g., file and original stdout)
class StreamTee(io.TextIOBase):
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
        # Ensure streams handle encoding correctly if needed, but rely on underlying streams mostly
        self._encoding = getattr(stream1, 'encoding', getattr(stream2, 'encoding', 'utf-8'))
        self._errors = getattr(stream1, 'errors', getattr(stream2, 'errors', 'strict'))

    @property
    def encoding(self):
        # Try to get encoding from the primary stream (usually the original sys.stdout/err)
        return getattr(self.stream1, 'encoding', self._encoding)

    @property
    def errors(self):
        return getattr(self.stream1, 'errors', self._errors)

    def write(self, data):
        # Write to both streams, return value like standard write
        written1 = -1
        written2 = -1
        try:
            written1 = self.stream1.write(data)
        except Exception as e:
            print(f"StreamTee Error writing to stream1: {e}", file=sys.__stderr__) # Use original stderr for logging errors
        try:
            written2 = self.stream2.write(data)
        except Exception as e:
            print(f"StreamTee Error writing to stream2: {e}", file=sys.__stderr__)
        # Return the number of characters written to the primary stream, or an average/max if needed
        return written1 if written1 != -1 else written2

    def flush(self):
        # Flush both streams
        try:
            self.stream1.flush()
        except Exception as e:
            print(f"StreamTee Error flushing stream1: {e}", file=sys.__stderr__)
        try:
            self.stream2.flush()
        except Exception as e:
            print(f"StreamTee Error flushing stream2: {e}", file=sys.__stderr__)

    def isatty(self):
        # Important for some libraries checking if output is a terminal
        return getattr(self.stream1, 'isatty', lambda: False)()

# Context manager to handle redirection
class Logger(object):
    def __init__(self, filename="Default.log", rank=0):
        self.rank = rank
        # Append mode to avoid overwriting on restarts
        self.log_file = open(filename, 'a', encoding='utf-8')
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        # Prefix output lines with rank information
        self.stdout_tee = StreamTee(self.original_stdout, self.log_file)
        self.stderr_tee = StreamTee(self.original_stderr, self.log_file)

    def __enter__(self):
        # Redirect stdout and stderr
        sys.stdout = self.stdout_tee
        sys.stderr = self.stderr_tee
        print(f"--- Rank {self.rank} Logging Started: Output redirected to console and {self.log_file.name} ---", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original streams
        print(f"--- Rank {self.rank} Logging Stopping: Restoring original stdout/stderr ---", flush=True)
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        # Close the log file
        self.log_file.close()
        # If an exception occurred, print it to the original stderr after restoring
        if exc_type is not None:
             print(f"Rank {self.rank}: Exception occurred within Logger context:", file=self.original_stderr)
             import traceback
             traceback.print_exception(exc_type, exc_val, exc_tb, file=self.original_stderr)

# === Additions End ===


# --- Dataset Definition ---
class GraphLatentDataset(Dataset):
    """
    Loads pre-computed graph Data objects and finds their corresponding
    VAE latent tensors based on paths stored within the graph objects.
    Filters out spots where the latent path is missing or invalid.
    """
    def __init__(self, graph_data_dir, latent_base_dir, rank=0, world_size=1):
        self.graph_data_dir = graph_data_dir
        self.latent_base_dir = latent_base_dir # Base directory where sample latent folders exist
        self.items = [] # List of tuples: (graph_path, node_idx_in_graph)

        if not os.path.isdir(graph_data_dir):
            raise FileNotFoundError(f"Graph data directory not found: {graph_data_dir}")

        all_graph_files = sorted(glob.glob(os.path.join(graph_data_dir, "*_graph.pt")))

        if rank == 0:
            print(f"Found {len(all_graph_files)} graph files in {graph_data_dir}")

        if not all_graph_files:
            raise FileNotFoundError(f"No graph files (.pt) found in {graph_data_dir}")

        # Split graph files across ranks for scanning
        num_files_per_rank = math.ceil(len(all_graph_files) / world_size)
        start_idx = rank * num_files_per_rank
        end_idx = min(start_idx + num_files_per_rank, len(all_graph_files))
        graphs_for_this_rank = all_graph_files[start_idx:end_idx]

        if rank == 0: print("Scanning graphs for valid latent paths...")
        pbar = tqdm(graphs_for_this_rank, desc=f"Rank {rank} Scanning", disable=(rank!=0))
        valid_spots_count = 0
        skipped_count = 0
        for graph_path in pbar:
            try:
                # Load only metadata quickly first if possible, or load full graph
                # For simplicity, load full graph here
                graph_data = torch.load(graph_path, map_location='cpu')
                if not hasattr(graph_data, 'latent_paths') or not hasattr(graph_data, 'num_nodes'):
                     if rank == 0: print(f"Warning: Graph {os.path.basename(graph_path)} missing 'latent_paths' or 'num_nodes'. Skipping.")
                     skipped_count += 1
                     continue

                has_valid_node = False
                for node_idx in range(graph_data.num_nodes):
                    latent_path = graph_data.latent_paths[node_idx]
                    # Check if path exists and is not None
                    if latent_path and os.path.exists(latent_path):
                        self.items.append((graph_path, node_idx))
                        valid_spots_count += 1
                        has_valid_node = True
                if not has_valid_node:
                    skipped_count += 1

            except Exception as e:
                if rank == 0: print(f"Warning: Error loading/processing graph {os.path.basename(graph_path)}: {e}. Skipping.")
                skipped_count += 1

        print(f"Rank {rank}: Found {len(self.items)} valid spots (nodes with existing latent files) from {len(graphs_for_this_rank)} graphs scanned ({skipped_count} graphs skipped or had no valid nodes).")
        if len(self.items) == 0 and rank==0:
            print("\nERROR: No valid spots found across all processed graphs on Rank 0.")
            print("Check:")
            print(f"  - If '{graph_data_dir}' contains valid graph files.")
            print(f"  - If graphs contain the 'latent_paths' attribute.")
            print(f"  - If paths in 'latent_paths' point to existing files under '{self.latent_base_dir}'.") # Use self.latent_base_dir here
            print(f"  - Latent path format in preprocessing notebook matches actual filenames.")
            # Consider sys.exit(1) if this is critical

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        graph_path, node_idx = self.items[idx]

        try:
            # Load the graph data object
            graph_data = torch.load(graph_path, map_location='cpu') # Keep on CPU for collate

            # Get the specific latent path for this node
            latent_path = graph_data.latent_paths[node_idx]

            # Load the VAE latent tensor
            latent = torch.load(latent_path, map_location='cpu') # Keep on CPU for collate

            # Return latent, graph object, and the index of the node within that graph
            # The graph object will be automatically batched by PyG DataLoader
            return latent.float(), graph_data, torch.tensor(node_idx, dtype=torch.long)

        except Exception as e:
            # Log error minimally, DataLoader will likely skip this item if collate handles None
            # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: ERROR loading item {idx} (graph: {os.path.basename(graph_path)}, node: {node_idx}): {e}")
            return None, None, None # Return None to be filtered by collate_fn

# --- Collate Function ---
def pyg_collate_fn(batch):
    """
    Collate function for the GraphLatentDataset.
    Filters Nones, batches latents, graphs (using PyG Batch), and node indices.
    """
    # Filter out None entries resulting from loading errors
    original_size = len(batch)
    batch = [item for item in batch if item[0] is not None and item[1] is not None and item[2] is not None]
    if not batch:
        #print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Warning - Collate function received an entirely empty batch after filtering Nones (original size: {original_size}).")
        return None, None, None # Return None if the entire batch failed

    latents = torch.stack([item[0] for item in batch], dim=0)
    graph_list = [item[1] for item in batch]
    node_indices_in_graphs = torch.stack([item[2] for item in batch], dim=0) # Original indices

    # Batch graphs using PyG's Batch class
    try:
        graph_batch = Batch.from_data_list(graph_list)
    except Exception as e:
        # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Error during Batch.from_data_list: {e}")
        # print(f"Graph list causing error: {graph_list}") # Potentially large output
        return None, None, None # Handle error gracefully

    # We need the *global* node index in the batched graph corresponding
    # to the original node_indices_in_graphs for each item in the batch.
    # graph_batch.ptr[i] gives the starting index for graph i in the flattened node list.
    # The batch index for each graph is implicitly 0, 1, 2... for this collate structure.
    global_node_indices = []
    try:
        for i in range(len(batch)): # Iterate through items in the original batch
             start_node_idx_for_graph_i = graph_batch.ptr[i]
             original_node_idx = node_indices_in_graphs[i].item()
             global_idx = start_node_idx_for_graph_i + original_node_idx
             global_node_indices.append(global_idx)
        global_node_indices_tensor = torch.tensor(global_node_indices, dtype=torch.long)
    except Exception as e:
        # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Error calculating global node indices: {e}")
        return None, None, None

    return latents, graph_batch, global_node_indices_tensor


# --- Model Definitions ---

# GraphConditioner (using GPSConv - Corrected version)
class GraphConditioner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, heads, attn_dropout=0.1):
        super().__init__()
        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.output_lin = nn.Linear(hidden_dim, output_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_mpnn = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                dropout=attn_dropout,
                add_self_loops=False,
                concat=True
            )
            conv = GPSConv(
                channels=hidden_dim,
                conv=local_mpnn,
                heads=heads,
                dropout=attn_dropout, # General dropout for GPSConv
                norm='layer'
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

# Function to get UNet model
def get_unet_model(args):
    """Creates the UNet model based on args."""
    block_out_channels = tuple(map(int, args.unet_block_out_channels.split(',')))
    down_block_types = tuple(args.unet_down_block_types.split(','))
    up_block_types = tuple(args.unet_up_block_types.split(','))

    model = UNet2DConditionModel(
        sample_size=args.unet_sample_size,
        in_channels=args.unet_in_channels,
        out_channels=args.unet_out_channels,
        layers_per_block=2,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        cross_attention_dim=args.unet_cross_attention_dim,
    )
    return model

# Function to get VAE for sampling
def get_vae_for_sampling(args):
    """Loads the VAE model specified in args, intended ONLY for rank 0 sampling."""
    if not args.vae_model_path or not os.path.exists(args.vae_model_path):
         print("Rank 0: Warning: VAE path for sampling not found or not specified. Using default TAESD.")
         model_name = "madebyollin/taesd"
         try:
             vae = AutoencoderTiny.from_pretrained(model_name, torch_dtype=torch.float32)
             print(f"Rank 0: Loaded default VAE: {model_name}")
         except Exception as e:
             print(f"Rank 0: Error loading default VAE {model_name}: {e}. Sampling will fail.")
             return None
    else:
         print(f"Rank 0: Loading VAE for sampling from: {args.vae_model_path}")
         try:
             vae = AutoencoderTiny(in_channels=3, out_channels=3, latent_channels=args.unet_in_channels)
             vae.load_state_dict(torch.load(args.vae_model_path, map_location='cpu'))
             print("Rank 0: Successfully loaded VAE for sampling.")
         except Exception as e:
             print(f"Rank 0: Error loading VAE from {args.vae_model_path}: {e}. Sampling might fail.")
             return None

    if not hasattr(vae.config, 'scaling_factor') or vae.config.scaling_factor is None:
        vae.config.scaling_factor = 1.0 # Default for TAESD if missing
    return vae


# --- Sampling Function (Unchanged from previous version) ---
@torch.no_grad()
def sample_and_compare(conditioner, unet_module, vae, scheduler, device, args, global_step, writer):
    """Generates conditioned samples and compares them to originals using matplotlib. (Rank 0 only)"""
    conditioner.eval()
    unet_module.eval() # Use the unwrapped UNet module
    if vae is None:
        print("Rank 0: VAE not available for sampling.")
        conditioner.train()
        unet_module.train()
        return
    vae.to(device)
    vae.eval()

    # --- Get Sample Graphs, Nodes, and Ground Truth Latents ---
    num_samples = 8 # Reduced default for better plot visibility
    graph_files = glob.glob(os.path.join(args.graph_data_dir, "*_graph.pt"))
    if not graph_files:
         print("Rank 0: No graph files found for sampling.")
         conditioner.train(); unet_module.train(); vae.cpu()
         return

    sampled_graphs_data = []
    sampled_node_indices = []
    ground_truth_latents = []
    attempts = 0
    max_attempts = min(len(graph_files) * 10, 200)

    print(f"Rank 0: Attempting to sample {num_samples} graph nodes with valid latents...")
    while len(sampled_graphs_data) < num_samples and attempts < max_attempts:
        graph_path = random.choice(graph_files)
        try:
            g = torch.load(graph_path, map_location='cpu')
            valid_node_indices = [idx for idx, p in enumerate(g.latent_paths) if p and os.path.exists(p)]
            if valid_node_indices:
                 node_idx = random.choice(valid_node_indices)
                 latent_path = g.latent_paths[node_idx]
                 gt_latent = torch.load(latent_path, map_location='cpu')

                 sampled_graphs_data.append(g)
                 sampled_node_indices.append(node_idx)
                 ground_truth_latents.append(gt_latent.float())

        except Exception as load_err:
             # print(f"Rank 0: Minor error during sampling prep (graph: {os.path.basename(graph_path)}): {load_err}")
             pass
        attempts += 1

    if len(sampled_graphs_data) < num_samples:
         print(f"Rank 0: Could only load {len(sampled_graphs_data)}/{num_samples} valid pairs for sampling after {attempts} attempts.")
         if not sampled_graphs_data:
              conditioner.train(); unet_module.train(); vae.cpu()
              return
         num_samples = len(sampled_graphs_data) # Adjust num_samples if fewer were loaded

    sampling_graph_batch = Batch.from_data_list(sampled_graphs_data).to(device)
    sampling_global_node_indices = []
    for i in range(len(sampled_graphs_data)):
        start_node_idx = sampling_graph_batch.ptr[i]
        original_node_idx = sampled_node_indices[i]
        sampling_global_node_indices.append(start_node_idx + original_node_idx)
    sampling_global_node_indices = torch.tensor(sampling_global_node_indices, dtype=torch.long, device=device)
    ground_truth_latents_batch = torch.stack(ground_truth_latents, dim=0).to(device)

    # --- Generate Conditions ---
    print(f"Rank 0: Generating embeddings for {num_samples} conditions...")
    amp_enabled_sampling = (args.mixed_precision in ["bf16", "fp16"])
    if amp_enabled_sampling:
         amp_dtype_sampling = torch.bfloat16 if args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    else:
         amp_dtype_sampling = torch.float32

    with autocast(enabled=amp_enabled_sampling, dtype=amp_dtype_sampling):
         all_node_embeddings = conditioner(sampling_graph_batch)
         spot_conditions = all_node_embeddings[sampling_global_node_indices]
         spot_conditions = spot_conditions.unsqueeze(1)

    # --- Diffusion Sampling Process ---
    latent_shape = (num_samples, args.unet_in_channels, args.unet_sample_size, args.unet_sample_size)
    noise = torch.randn(latent_shape, device=device, dtype=torch.float32)
    generated_latents = noise.clone()

    scheduler.set_timesteps(args.sampling_inference_steps)
    sampling_steps = tqdm(scheduler.timesteps, desc="Sampling Comparison", leave=False, disable=False)

    for t in sampling_steps:
        latent_model_input = scheduler.scale_model_input(generated_latents, t)
        timestep_tensor = torch.tensor([t] * generated_latents.shape[0], device=device).long()

        with autocast(enabled=amp_enabled_sampling, dtype=amp_dtype_sampling):
             noise_pred = unet_module(
                 sample=latent_model_input.to(amp_dtype_sampling),
                 timestep=timestep_tensor,
                 encoder_hidden_states=spot_conditions.to(amp_dtype_sampling)
             ).sample

        generated_latents = scheduler.step(noise_pred.float(), t, generated_latents).prev_sample

    # --- Decode Latents (Generated and Ground Truth) ---
    scale_factor = getattr(vae.config, "scaling_factor", 1.0)
    generated_latents_to_decode = generated_latents / scale_factor
    gt_latents_to_decode = ground_truth_latents_batch / scale_factor

    with torch.no_grad():
        with autocast(enabled=amp_enabled_sampling, dtype=amp_dtype_sampling):
            decoded_generated = vae.decode(generated_latents_to_decode.to(amp_dtype_sampling)).sample
            decoded_originals = vae.decode(gt_latents_to_decode.to(amp_dtype_sampling)).sample

    # Post-process images to [0, 1] range for plotting
    generated_images = (decoded_generated.clamp(-1, 1) + 1) / 2
    original_images = (decoded_originals.clamp(-1, 1) + 1) / 2
    generated_images = generated_images.cpu().float() # Move to CPU and ensure float
    original_images = original_images.cpu().float()

    # --- Create Comparison Plot with Matplotlib ---
    print(f"Rank 0: Creating comparison plot for {num_samples} pairs...")
    plt.switch_backend('Agg') # Use non-GUI backend

    nrows = num_samples
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes_flat = axes.flatten() if nrows > 1 else axes

    for i in range(num_samples):
        ax_gen = axes_flat[i * 2]
        img_gen = generated_images[i].permute(1, 2, 0).numpy()
        ax_gen.imshow(img_gen)
        ax_gen.set_title(f"Sample {i+1}: Generated", fontsize=10)
        ax_gen.axis('off')

        ax_orig = axes_flat[i * 2 + 1]
        img_orig = original_images[i].permute(1, 2, 0).numpy()
        ax_orig.imshow(img_orig)
        ax_orig.set_title(f"Sample {i+1}: Original", fontsize=10)
        ax_orig.axis('off')

    fig.suptitle(f'Generated vs Original Comparison (Step {global_step})', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Log and Save ---
    log_tag = 'Step_Comparison_Plot'
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    pil_img = PIL.Image.open(buf)
    plot_tensor = transforms.ToTensor()(pil_img)
    if writer: # Check if writer exists
      writer.add_image(log_tag, plot_tensor, global_step)
    buf.close()

    save_dir = os.path.join(args.log_dir, "step_samples_comparison")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"comparison_step_{global_step}.png")
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Rank 0: Saved comparison plot to {save_path}")
    plt.close(fig)

    # --- Cleanup ---
    conditioner.train()
    unet_module.train()
    vae.cpu()


# --- Checkpoint Utilities ---
def save_checkpoint(epoch, global_step, conditioner_module, unet_module, optimizer, lr_scheduler, scaler, args):
    """Saves conditioner, UNet, optimizer, etc. state (Rank 0 only). Includes global_step."""
    if dist.get_rank() != 0: return # Only save on rank 0

    checkpoint_name = (
        f"{args.checkpoint_filename_prefix}_"
        f"ep{epoch+1}_step{global_step}_bs{args.batch_size_per_gpu}x{dist.get_world_size()}_"
        f"lr{args.lr}_acc{args.accumulation_steps}.pt"
    )
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step, # Save global step
        'conditioner_state_dict': conditioner_module.state_dict(),
        'unet_state_dict': unet_module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'args': vars(args)
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Rank 0: Checkpoint saved at epoch {epoch+1} / step {global_step} to {checkpoint_path}")

def load_checkpoint(conditioner_module, unet_module, optimizer, lr_scheduler, scaler, device, args):
    """Loads checkpoint state into models, optimizer, etc. (All Ranks). Returns start_epoch, start_step."""
    start_epoch = 0
    start_step = 0 # Initialize start step
    checkpoint_path = None

    prefix = args.checkpoint_filename_prefix
    search_pattern = os.path.join(args.checkpoint_dir, f"{prefix}_ep*_step*.pt") # Include step in pattern
    checkpoints = []
    if os.path.isdir(args.checkpoint_dir):
        checkpoints = glob.glob(search_pattern)

    if checkpoints:
        # Find the checkpoint with the highest global step
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.split('_step')[-1].split('_')[0]))
        checkpoint_path = latest_checkpoint
        if dist.get_rank() == 0: print(f"Found latest checkpoint by step: {checkpoint_path}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        if dist.get_rank() == 0: print(f"Attempting to load checkpoint from {checkpoint_path}")
        try:
            map_location = {'cuda:0': f'cuda:{device.index}'} # Map to current device
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

            # Load Conditioner state
            if 'conditioner_state_dict' in checkpoint:
                conditioner_module.load_state_dict(checkpoint['conditioner_state_dict'])
            else:
                print(f"Rank {dist.get_rank()}: Warning: Conditioner state not found in checkpoint.")

            # Load UNet state
            if 'unet_state_dict' in checkpoint:
                unet_module.load_state_dict(checkpoint['unet_state_dict'])
            else: # Fallback logic for UNet state
                 if args.pretrained_unet_path and args.pretrained_unet_path != "None" and os.path.exists(args.pretrained_unet_path):
                     print(f"Rank {dist.get_rank()}: UNet state missing in main checkpoint, loading from --pretrained_unet_path: {args.pretrained_unet_path}")
                     try:
                        unet_checkpoint = torch.load(args.pretrained_unet_path, map_location=map_location)
                        state_dict_key = next((k for k in ['unet_state_dict', 'state_dict'] if k in unet_checkpoint), None)
                        if state_dict_key:
                            unet_module.load_state_dict(unet_checkpoint[state_dict_key])
                        else: # Assume it's just the state dict
                            unet_module.load_state_dict(unet_checkpoint)
                        print(f"Rank {dist.get_rank()}: Loaded UNet from --pretrained_unet_path.")
                     except Exception as e_unet_load:
                        print(f"Rank {dist.get_rank()}: ERROR loading UNet from --pretrained_unet_path: {e_unet_load}. UNet starts fresh.")
                 else:
                     print(f"Rank {dist.get_rank()}: Warning: UNet state not found in checkpoint and no valid pretrained path provided. UNet starts fresh.")

            # Load Optimizer, Scheduler, Scaler state (optional, allows exact resume)
            if 'optimizer_state_dict' in checkpoint and optimizer:
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            elif optimizer: print(f"Rank {dist.get_rank()}: Optimizer state not found. Optimizer starts fresh.")

            if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                 lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            elif lr_scheduler: print(f"Rank {dist.get_rank()}: LR Scheduler state not found.")

            if scaler and 'scaler_state_dict' in checkpoint:
                 scaler.load_state_dict(checkpoint['scaler_state_dict'])
            elif scaler: print(f"Rank {dist.get_rank()}: GradScaler state not found.")

            start_epoch = checkpoint.get('epoch', 0) + 1 # Start from epoch AFTER the saved one
            start_step = checkpoint.get('global_step', 0) # Resume from the saved step
            print(f"Rank {dist.get_rank()}: Resuming training from Epoch {start_epoch}, Step {start_step + 1}") # Adjust epoch print

        except Exception as e:
            print(f"Rank {dist.get_rank()}: ERROR loading checkpoint {checkpoint_path}: {e}")
            print(f"Rank {dist.get_rank()}: Starting training from scratch or only with pretrained UNet if provided.")
            start_epoch = 0
            start_step = 0
            # Fallback UNet loading (if checkpoint load failed but pretrained exists)
            if args.pretrained_unet_path and args.pretrained_unet_path != "None" and os.path.exists(args.pretrained_unet_path):
                 print(f"Rank {dist.get_rank()}: Attempting to load UNet from --pretrained_unet_path as fallback.")
                 try:
                     map_location = {'cuda:0': f'cuda:{device.index}'}
                     unet_checkpoint = torch.load(args.pretrained_unet_path, map_location=map_location)
                     state_dict_key = next((k for k in ['unet_state_dict', 'state_dict'] if k in unet_checkpoint), None)
                     if state_dict_key:
                         unet_module.load_state_dict(unet_checkpoint[state_dict_key])
                     else:
                         unet_module.load_state_dict(unet_checkpoint)
                     print(f"Rank {dist.get_rank()}: Successfully loaded UNet from fallback path.")
                 except Exception as e_unet:
                     print(f"Rank {dist.get_rank()}: ERROR loading UNet from fallback path {args.pretrained_unet_path}: {e_unet}")

    else:
        print(f"Rank {dist.get_rank()}: No checkpoint found matching pattern '{search_pattern}'.")
        # Load pretrained UNet if specified and no checkpoint exists
        if args.pretrained_unet_path and args.pretrained_unet_path != "None" and os.path.exists(args.pretrained_unet_path):
            print(f"Rank {dist.get_rank()}: Loading UNet from --pretrained_unet_path: {args.pretrained_unet_path}")
            try:
                map_location = {'cuda:0': f'cuda:{device.index}'}
                unet_checkpoint = torch.load(args.pretrained_unet_path, map_location=map_location)
                state_dict_key = next((k for k in ['unet_state_dict', 'state_dict'] if k in unet_checkpoint), None)
                if state_dict_key:
                    unet_module.load_state_dict(unet_checkpoint[state_dict_key])
                else:
                    unet_module.load_state_dict(unet_checkpoint)
                print(f"Rank {dist.get_rank()}: Loaded UNet from --pretrained_unet_path.")
            except Exception as e:
                print(f"Rank {dist.get_rank()}: ERROR loading UNet from --pretrained_unet_path: {e}. Starting UNet from scratch.")
        else:
            print(f"Rank {dist.get_rank()}: No pretrained UNet path provided. Starting UNet from scratch.")
        print(f"Rank {dist.get_rank()}: Starting training from scratch (Epoch 1, Step 1). Conditioner also starts from scratch.")
        start_epoch = 0
        start_step = 0

    return start_epoch, start_step


# +++ 新增：辅助日志函数 (保持不变) +++
def log_tensor_stats(tensor, name="Tensor", rank=0, print_stats=True):
    """打印 Tensor 的详细信息"""
    if not dist.is_initialized() or dist.get_rank() == rank: # 只在指定 rank 打印
        if tensor is None:
            print(f"[Rank {rank}] Stats for {name}: Tensor is None")
            return

        print(f"[Rank {rank}] Stats for {name}:")
        print(f"  - Shape: {tensor.shape}")
        print(f"  - Dtype: {tensor.dtype}")
        print(f"  - Device: {tensor.device}")

        try: # 添加 try-except 以防检查本身出错
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            print(f"  - Has NaN: {has_nan}")
            print(f"  - Has Inf: {has_inf}")

            # 仅对浮点型、非空、无 NaN/Inf 的 Tensor 计算统计值
            if print_stats and tensor.numel() > 0 and tensor.is_floating_point() and not has_nan and not has_inf:
                try:
                    tensor_float = tensor.float() # 转换为 float 计算统计
                    print(f"  - Min: {tensor_float.min().item():.6e}")
                    print(f"  - Max: {tensor_float.max().item():.6e}")
                    print(f"  - Mean: {tensor_float.mean().item():.6e}")
                    print(f"  - Std: {tensor_float.std().item():.6e}")
                except Exception as e_stat:
                    print(f"  - Error calculating stats: {e_stat}")
            elif not tensor.is_floating_point():
                 print("  - (Not a floating point tensor, skipping min/max/mean/std)")
            print("-" * 30)
        except Exception as e_check:
             print(f"  - Error during NaN/Inf check: {e_check}")
             print("-" * 30)
# +++ 结束新增 +++


# --- Main Training Function ---
# train_condition_diffusion_ddp.py
# ... (之前的 Imports, 函数定义, Helper 函数 log_tensor_stats 都保持不变) ...

# --- Main Training Function ---
def train_diffusion_model(rank, world_size, local_rank, args):
    """Main DDP training loop for conditioned diffusion."""
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (rank == 0)

    # +++ 第1处修改：定义需要深入检查的 global_step +++
    check_gradients_step = 2180 # 当 global_step 是这个值时，检查每个累积批次的梯度
    # detailed_log_trigger_steps = {2179} # 不再需要这个，或者可以注释掉
    # +++ 结束修改 +++

    # --- Mixed Precision Setup ---
    amp_enabled = (args.mixed_precision in ["bf16", "fp16"])
    amp_dtype = torch.float32
    if amp_enabled:
        if args.mixed_precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            if is_main_process: print("Using bfloat16 mixed precision.")
        elif args.mixed_precision == "fp16":
             amp_dtype = torch.float16
             if is_main_process: print("Using float16 mixed precision.")
        else:
             amp_dtype = torch.float16
             if is_main_process: print("Warning: bf16 selected but not supported, using float16 mixed precision.")
    else:
        if is_main_process: print("Mixed precision disabled. Using float32.")
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))

    # --- TensorBoard (Rank 0 Only) ---
    writer = None
    if is_main_process:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log_dir = os.path.join(args.log_dir, f"run_{current_time}")
        os.makedirs(run_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_log_dir)
        print(f"TensorBoard logs: {run_log_dir}")
        try:
            with open(os.path.join(run_log_dir, "config_args.txt"), "w") as f:
                 import json
                 json.dump(vars(args), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config args: {e}")

    # --- Dataset and DataLoader ---
    if is_main_process: print("Initializing dataset...")
    try:
        dataset = GraphLatentDataset(args.graph_data_dir, args.latent_dir, rank, world_size)
        if len(dataset) == 0:
             if rank == 0: print("ERROR: Dataset is empty after filtering for valid spots. Exiting.")
             dist.barrier() # Wait for others before exiting
             cleanup_ddp()
             sys.exit(1)
    except FileNotFoundError as e:
        if rank == 0: print(f"ERROR: {e}. Exiting.")
        dist.barrier()
        cleanup_ddp()
        sys.exit(1)
    except Exception as e_data:
         if rank == 0: print(f"ERROR during dataset initialization: {e_data}. Exiting.")
         dist.barrier()
         cleanup_ddp()
         sys.exit(1)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = PyGDataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=pyg_collate_fn
    )
    if is_main_process: print(f"DataLoader initialized with {args.num_workers} workers per GPU.")

    # --- Models ---
    if is_main_process: print("Initializing models...")
    conditioner = GraphConditioner(
        input_dim=args.conditioner_input_dim,
        hidden_dim=args.conditioner_hidden_dim,
        output_dim=args.conditioner_output_dim,
        num_layers=args.conditioner_n_layers,
        heads=args.conditioner_n_heads,
        attn_dropout=args.conditioner_attn_dropout
    )
    unet = get_unet_model(args)
    vae = get_vae_for_sampling(args) if is_main_process else None # VAE only on Rank 0 for sampling
    conditioner.to(device)
    unet.to(device)

    # --- Optimizer and Scheduler ---
    combined_params = list(conditioner.parameters()) + list(unet.parameters())
    optimizer = optim.AdamW(combined_params, lr=args.lr, weight_decay=1e-4)
    num_batches_per_epoch_approx = len(dataloader)
    num_update_steps_per_epoch = math.ceil(num_batches_per_epoch_approx / args.accumulation_steps)
    total_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = None
    if total_train_steps > 0:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_train_steps, eta_min=1e-7
        )
        if is_main_process:
            print(f"LR Scheduler: Cosine Annealing, T_max={total_train_steps} steps")
    else:
         if is_main_process: print("Warning: Zero training steps detected. LR scheduler disabled.")
    if is_main_process:
        print(f"Optimizer: AdamW, LR: {args.lr}")

    # --- Load Checkpoint ---
    start_epoch, start_step = load_checkpoint(conditioner, unet, optimizer, lr_scheduler, scaler, device, args)

    # --- Wrap Models with DDP ---
    if is_main_process: print("Wrapping models with DDP...")
    conditioner = DDP(conditioner, device_ids=[local_rank], find_unused_parameters=True)
    unet = DDP(unet, device_ids=[local_rank], find_unused_parameters=True)

    # --- Training Loop ---
    global_step = start_step
    total_start_time = time.time()
    if is_main_process:
         print("\n" + "="*30 + " STARTING TRAINING " + "="*30)
         print(f"  Target Epochs: {args.epochs}")
         print(f"  Start Epoch: {start_epoch}") # Use loaded start_epoch directly
         print(f"  Start Step: {start_step + 1}")
         print(f"  World Size: {world_size}")
         print(f"  Batch Size Per GPU: {args.batch_size_per_gpu}")
         print(f"  Accumulation Steps: {args.accumulation_steps}")
         print(f"  Effective Total Batch Size: {args.batch_size_per_gpu * world_size * args.accumulation_steps}")
         print(f"  Mixed Precision: {args.mixed_precision} (dtype: {amp_dtype})")
         print(f"  Dataset Size (Rank 0 Estimate): {len(dataset)}")
         print(f"  Steps per Epoch (per GPU, approx): ~{num_update_steps_per_epoch}")
         print(f"  Total Estimated Train Steps: {total_train_steps}")
         print(f"  Log Interval (Steps): {args.log_interval}")
         print(f"  Sample Interval (Steps): {args.sample_interval_steps}")
         print(f"  Save Interval (Epochs): {args.save_interval}")
         print(f"  Gradient Check Step: {check_gradients_step}") # Print the check step
         print("="*79 + "\n")

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.scheduler_train_timesteps)

    # --- Resume LR Scheduler State ---
    if lr_scheduler and start_step > 0:
        if is_main_process: print(f"Fast-forwarding LR scheduler to step {start_step}...")
        # Simulate optimizer steps to advance scheduler
        # Temporarily set dummy gradients to allow optimizer.step()
        for p_group in optimizer.param_groups:
            for p in p_group['params']:
                if p.grad is None:
                    # Ensure dummy grad is on the correct device and dtype
                    p.grad = torch.zeros_like(p.data) # Use p.data to avoid requires_grad issues
        for _ in range(start_step):
            optimizer.step()
            lr_scheduler.step()
        optimizer.zero_grad()
        if is_main_process: print(f"LR Scheduler state advanced. Current LR: {optimizer.param_groups[0]['lr']:.2e}")

    # ============================= MAIN TRAINING LOOP =============================
    for epoch in range(start_epoch, args.epochs):
        conditioner.train()
        unet.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_samples_processed = 0
        epoch_start_time = time.time()

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}", # Display current epoch correctly
            disable=not is_main_process,
            leave=True
        )
        optimizer.zero_grad() # Reset grads at the start of epoch

        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None or batch_data[0] is None:
                 continue

            latents, graph_batch, global_node_indices = batch_data
            latents = latents.to(device, non_blocking=True)
            graph_batch = graph_batch.to(device, non_blocking=True)
            global_node_indices = global_node_indices.to(device, non_blocking=True)
            current_batch_size = latents.shape[0]
            epoch_samples_processed += current_batch_size * world_size

            # --- 第2处修改：移除大部分详细日志 ---
            is_last_batch = (batch_idx + 1) == len(dataloader)
            is_optimizer_step_batch = (batch_idx + 1) % args.accumulation_steps == 0 or is_last_batch
            # --- 结束第2处修改 ---

            # --- Sample noise and timesteps ---
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (current_batch_size,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # --- Autocast block ---
            try:
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    # 1. Generate Condition
                    all_node_embeddings = conditioner(graph_batch) # DDP Forward
                    # +++ 第3处修改(1/4): 添加 Conditioner 输出检查 +++
                    if torch.isnan(all_node_embeddings).any() or torch.isinf(all_node_embeddings).any():
                         print(f"!!!!!!!! Rank {rank} Step {global_step} Batch {batch_idx}: NaN/Inf in Conditioner output !!!!!!!!!!")
                         # dist.barrier()
                         # sys.exit(f"Rank {rank} aborted due to NaN/Inf in Conditioner.")
                    # +++ 结束添加 +++

                    # 2. Select conditions
                    spot_conditions = all_node_embeddings[global_node_indices]
                    spot_conditions = spot_conditions.unsqueeze(1)
                    # +++ 第3处修改(2/4): 添加 spot_conditions 检查 +++
                    if torch.isnan(spot_conditions).any() or torch.isinf(spot_conditions).any():
                         print(f"!!!!!!!! Rank {rank} Step {global_step} Batch {batch_idx}: NaN/Inf in spot_conditions !!!!!!!!!!")
                    # +++ 结束添加 +++

                    # 3. Predict noise using UNet
                    noise_pred = unet( # DDP Forward
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=spot_conditions
                    ).sample
                    # +++ 第3处修改(3/4): 添加 UNet 输出检查 +++
                    if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                         print(f"!!!!!!!! Rank {rank} Step {global_step} Batch {batch_idx}: NaN/Inf in UNet output !!!!!!!!!!")
                    # +++ 结束添加 +++

                    # 4. Calculate loss
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                    # +++ 第3处修改(4/4): 添加 Loss 检查 +++
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                         print(f"!!!!!!!! Rank {rank} Step {global_step} Batch {batch_idx}: Loss is NaN/Inf !!!!!!!!!!")
                         dist.barrier()
                         sys.exit(f"Rank {rank} aborted due to NaN/Inf loss at step {global_step}, batch {batch_idx}.")
                    # +++ 结束添加 +++

                    loss_unscaled = loss.item()
                    loss = loss / args.accumulation_steps # Scale loss for accumulation

            except Exception as forward_err:
                 print(f"!!!!!!!! Rank {rank} Step {global_step} Batch {batch_idx}: ERROR during forward pass or loss calculation !!!!!!!!!!")
                 print(f"Error type: {type(forward_err)}")
                 print(f"Error message: {forward_err}")
                 import traceback
                 traceback.print_exc()
                 dist.barrier()
                 sys.exit(f"Rank {rank} aborted due to error in forward pass at step {global_step}, batch {batch_idx}.")

            # --- Backpropagate ---
            try:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward() # DDP handles gradient sync here

                # +++ 第4处修改：添加特定步骤的梯度检查 +++
                if global_step == check_gradients_step:
                    # --- Gradient Check Logic ---
                    print(f"--- [Rank {rank}] Grad Check after backward for Step {global_step}, Batch {batch_idx} ---")
                    nan_grads_found_this_batch = False
                    inf_grads_found_this_batch = False
                    # Check Conditioner gradients
                    for name, param in conditioner.module.named_parameters(): # Access .module under DDP
                        if param.grad is not None:
                            try:
                                if torch.isnan(param.grad).any():
                                    print(f"!!! [Rank {rank}] Step {global_step} Batch {batch_idx}: NaN grad in Conditioner param {name} !!!")
                                    nan_grads_found_this_batch = True
                                if torch.isinf(param.grad).any():
                                    print(f"!!! [Rank {rank}] Step {global_step} Batch {batch_idx}: Inf grad in Conditioner param {name} !!!")
                                    inf_grads_found_this_batch = True
                            except Exception as grad_check_e:
                                print(f"!!! [Rank {rank}] Step {global_step} Batch {batch_idx}: Error checking grad for Conditioner param {name}: {grad_check_e} !!!")
                    # Check UNet gradients
                    for name, param in unet.module.named_parameters(): # Access .module under DDP
                         if param.grad is not None:
                            try:
                                if torch.isnan(param.grad).any():
                                    print(f"!!! [Rank {rank}] Step {global_step} Batch {batch_idx}: NaN grad in UNet param {name} !!!")
                                    nan_grads_found_this_batch = True
                                if torch.isinf(param.grad).any():
                                    print(f"!!! [Rank {rank}] Step {global_step} Batch {batch_idx}: Inf grad in UNet param {name} !!!")
                                    inf_grads_found_this_batch = True
                            except Exception as grad_check_e:
                                 print(f"!!! [Rank {rank}] Step {global_step} Batch {batch_idx}: Error checking grad for UNet param {name}: {grad_check_e} !!!")

                    if nan_grads_found_this_batch or inf_grads_found_this_batch:
                        print(f"!!!!!!!! [Rank {rank}] Step {global_step} Batch {batch_idx}: NaN/Inf gradients detected after backward! Likely cause of NCCL hang. !!!!!!!!!!")
                        # Optional: Exit immediately after finding bad gradients
                        # dist.barrier()
                        # sys.exit(f"Rank {rank} aborted due to NaN/Inf gradients detected at step {global_step}, batch {batch_idx}.")
                    else:
                         print(f"--- [Rank {rank}] Grad Check OK for Step {global_step}, Batch {batch_idx} ---")
                # +++ 结束添加 +++

            except Exception as backward_err:
                 print(f"!!!!!!!! Rank {rank} Step {global_step} Batch {batch_idx}: ERROR during backward() !!!!!!!!!!")
                 print(f"Error type: {type(backward_err)}")
                 print(f"Error message: {backward_err}")
                 import traceback
                 traceback.print_exc()
                 # Attempt barrier before exiting
                 try:
                     dist.barrier()
                 except Exception as barrier_err:
                     print(f"Rank {rank} exception during barrier after backward error: {barrier_err}")
                 sys.exit(f"Rank {rank} aborted due to error in backward() at step {global_step}, batch {batch_idx}.")

            # --- Accumulate loss ---
            epoch_loss += loss_unscaled * current_batch_size

            # --- Optimizer Step after accumulation ---
            if is_optimizer_step_batch:
                grad_norm = 0.0 # Initialize grad_norm

                # Gradient Clipping
                try:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    # Important: Clip gradients for the *combined* parameters
                    # Access parameters through .module when wrapped with DDP
                    combined_params_for_clip = list(conditioner.module.parameters()) + list(unet.module.parameters())
                    # Filter out parameters without gradients before clipping
                    params_to_clip = [p for p in combined_params_for_clip if p.grad is not None]
                    if params_to_clip: # Only clip if there are grads to clip
                        grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0).item()
                except Exception as clip_err:
                     print(f"!!!!!!!! Rank {rank} Step {global_step}: ERROR during gradient clipping !!!!!!!!!!")
                     print(f"Error message: {clip_err}")
                     # Consider exiting

                # Optimizer step
                try:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                except Exception as opt_err:
                     print(f"!!!!!!!! Rank {rank} Step {global_step}: ERROR during optimizer.step() !!!!!!!!!!")
                     print(f"Error message: {opt_err}")
                     # Consider exiting

                # Zero gradients AFTER step and update
                optimizer.zero_grad(set_to_none=True)

                # Step LR scheduler
                if lr_scheduler:
                    lr_scheduler.step()

                # Increment global step counter ONLY on optimizer step
                global_step += 1

                # --- Logging & Sampling (Rank 0 Only) ---
                if is_main_process:
                    if writer and global_step % args.log_interval == 0: # Check if writer exists
                        writer.add_scalar('Train/Batch_Loss_Rank0', loss_unscaled, global_step)
                        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                        writer.add_scalar('Train/Gradient_Norm', grad_norm, global_step) # Log grad_norm
                        if scaler.is_enabled():
                            writer.add_scalar('Train/GradScaler_Scale', scaler.get_scale(), global_step)

                    # <---- 中断处在此之后 ---->
                    if global_step > 0 and global_step % args.sample_interval_steps == 0:
                         print(f"\n--- Generating comparison samples at Step {global_step} ---")
                         sample_and_compare(
                             conditioner.module, # Pass unwrapped module
                             unet.module,        # Pass unwrapped module
                             vae,
                             noise_scheduler,
                             device,
                             args,
                             global_step,
                             writer
                         )
                         conditioner.train() # Ensure model is back in train mode
                         unet.train()
            # --- End of Optimizer Step Block ---

            # --- Update Progress Bar ---
            if is_main_process:
                progress_bar.set_postfix(
                    loss=f"{loss_unscaled:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    step=global_step # Show current global step
                )
        # --- End of Batch Loop ---

        # --- End of Epoch ---
        # Calculate average loss across all samples processed in the epoch on Rank 0
        avg_epoch_loss_rank0 = epoch_loss / epoch_samples_processed if epoch_samples_processed > 0 else 0.0

        # Gather average loss across all ranks
        avg_loss_tensor_data = torch.tensor([avg_epoch_loss_rank0 * epoch_samples_processed, epoch_samples_processed], device=device, dtype=torch.float64) # Sum of losses and count
        dist.all_reduce(avg_loss_tensor_data, op=dist.ReduceOp.SUM)
        total_epoch_loss_all = avg_loss_tensor_data[0].item()
        total_samples_all = avg_loss_tensor_data[1].item()
        avg_epoch_loss_all = total_epoch_loss_all / total_samples_all if total_samples_all > 0 else 0.0

        epoch_duration = time.time() - epoch_start_time
        total_elapsed = time.time() - total_start_time

        if is_main_process:
            print(f"\nEpoch {epoch+1} completed. Avg Loss (All Ranks): {avg_epoch_loss_all:.6f}")
            print(f"Time: {str(datetime.timedelta(seconds=int(epoch_duration)))}, Total: {str(datetime.timedelta(seconds=int(total_elapsed)))}")
            if writer:
                writer.add_scalar('Train/Epoch_Avg_Loss_All', avg_epoch_loss_all, epoch + 1)
                writer.add_scalar('Train/Epoch_Duration_sec', epoch_duration, epoch + 1)

            # Save Checkpoint Periodically by Epoch
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                save_checkpoint(
                    epoch, global_step, conditioner.module, unet.module,
                    optimizer, lr_scheduler, scaler, args
                 )

        dist.barrier() # Sync all processes before next epoch
    # --- End of Epoch Loop ---

    # --- End of Training ---
    if is_main_process:
        print(f"\n" + "="*28 + " CONDITIONED TRAINING FINISHED " + "="*28)
        print(f"Total time: {str(datetime.timedelta(seconds=int(time.time() - total_start_time)))}")
        print("Saving final model state...")
        save_checkpoint(args.epochs - 1, global_step, conditioner.module, unet.module, optimizer, lr_scheduler, scaler, args)
        if writer:
            writer.close()
            print("TensorBoard writer closed.")
    dist.barrier()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Conditioned Diffusion Model Training Script")

    # --- Add Arguments (Copied from notebook cell) ---
    parser.add_argument('--graph_data_dir', type=str, required=True)
    parser.add_argument('--latent_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--vae_model_path', type=str, required=True)
    parser.add_argument('--vae_sd_version', type=str, default='v2.1')
    parser.add_argument('--pretrained_unet_path', type=str, default="None")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size_per_gpu', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--mixed_precision', type=str, default='bf16', choices=['bf16', 'fp16', 'no'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pca_n_comps', type=int, default=50)
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
    parser.add_argument('--scheduler_train_timesteps', type=int, default=1000)
    parser.add_argument('--sampling_inference_steps', type=int, default=50)
    parser.add_argument('--checkpoint_filename_prefix', type=str, default="cond_unet_gps")
    parser.add_argument('--log_interval', type=int, default=20, help="Log basic stats every N steps")
    parser.add_argument('--sample_interval_steps', type=int, default=200, help="Generate comparison samples every N steps")
    parser.add_argument('--save_interval', type=int, default=1, help="Save checkpoint every N epochs")

    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Initialize DDP ---
    rank, world_size, local_rank = setup_ddp()

    # --- Simple validation for required args ---
    required_for_model = [
        args.conditioner_input_dim, args.conditioner_hidden_dim, args.conditioner_output_dim,
        args.conditioner_n_layers, args.conditioner_n_heads, args.conditioner_attn_dropout,
        args.unet_sample_size, args.unet_in_channels, args.unet_out_channels,
        args.unet_block_out_channels, args.unet_down_block_types, args.unet_up_block_types,
        args.unet_cross_attention_dim
    ]
    if any(val is None for val in required_for_model):
        if rank == 0: print("ERROR: One or more required model configuration arguments are missing.")
        dist.barrier()
        sys.exit(1)
    

    # --- Determine and Share Log Directory for this Run ---
    run_log_dir = None
    if rank == 0:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log_dir = os.path.join(args.log_dir, f"run_{current_time}")
        try:
            os.makedirs(run_log_dir, exist_ok=True)
            print(f"Rank 0: Created run log directory: {run_log_dir}")
        except OSError as e:
            print(f"Rank 0: ERROR creating log directory {run_log_dir}: {e}", file=sys.stderr)
            # Decide how to handle: maybe try a default or exit
            run_log_dir = args.log_dir # Fallback, might cause issues if permissions are wrong
        # Prepare the directory path to be broadcasted
        log_dir_list = [run_log_dir]
    else:
        log_dir_list = [None] # Placeholder for other ranks

    # Broadcast the directory name from rank 0 to all other ranks
    if world_size > 1:
        try:
            dist.broadcast_object_list(log_dir_list, src=0)
        except Exception as e:
             print(f"Rank {rank}: ERROR during broadcast_object_list: {e}", file=sys.__stderr__) # Use original stderr
             # Attempt to cleanup and exit if broadcast fails
             cleanup_ddp()
             sys.exit(1)

    run_log_dir = log_dir_list[0] # All ranks now have the same path

    if run_log_dir is None:
         print(f"Rank {rank}: ERROR - Failed to get run_log_dir from Rank 0. Exiting.", file=sys.__stderr__)
         cleanup_ddp()
         sys.exit(1)

    # Ensure all ranks wait until rank 0 has likely finished creating the directory
    if world_size > 1:
        dist.barrier()

    # --- Setup File Logging for Each Rank ---
    log_file_path = os.path.join(run_log_dir, f"rank_{rank}_output.log")

    # Use the Logger context manager to redirect stdout/stderr
    with Logger(log_file_path, rank):
        try:
            # --- Start Training (pass the determined run_log_dir) ---
            train_diffusion_model(rank, world_size, local_rank, args, run_log_dir) # Pass run_log_dir
        except KeyboardInterrupt:
            # Logger will capture the "KeyboardInterrupt received" message if printed before cleanup
            print(f"\nRank {rank}: KeyboardInterrupt received. Cleaning up...")
            # Cleanup will happen in the finally block
        except Exception as e:
            # The Logger's __exit__ will print the traceback to original stderr
            # And the exception details will be logged to the file via the redirected stderr
            print(f"Rank {rank}: Global exception handler caught error: {e}", file=sys.stderr) # Logged to file+console
            # Traceback is handled by Logger's __exit__ on original stderr
            pass # Exception is already logged, let finally run
        finally:
            # --- Cleanup DDP ---
            # DDP cleanup might print messages which should ideally NOT be logged,
            # so perform it outside the 'with Logger(...)' block if possible,
            # OR accept that its messages might go to the file.
            # Putting it here means cleanup messages go to original console only.
            pass # DDP cleanup happens after the Logger context closes.

    # === Modification End ===

    # DDP cleanup is now outside the logger context
    cleanup_ddp()
# === END OF SCRIPT ===