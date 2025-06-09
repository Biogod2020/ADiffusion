# train_diffusion_ddp.py
import os
import argparse
import random
import glob
import datetime
import time
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm # Use standard tqdm in script

from diffusers import AutoencoderTiny, UNet2DConditionModel, DDPMScheduler

# --- DDP Setup/Cleanup ---
def setup_ddp():
    """Initializes the distributed process group."""
    # assumes environment variables are set by torchrun/launch
    dist.init_process_group(backend="nccl") # NCCL is standard for NVIDIA GPUs
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    print(f"DDP Setup: Rank {rank}/{world_size}, Local Rank {local_rank}, Device cuda:{local_rank}")
    return rank, world_size, local_rank

def cleanup_ddp():
    """Destroys the distributed process group."""
    dist.destroy_process_group()
    print("DDP Cleanup Completed.")

# --- Latent Dataset (same as in notebook, but used here) ---
# train_diffusion_ddp.py

# --- Latent Dataset (Modified for recursive search) ---
class LatentDataset(Dataset):
    """Loads pre-computed latent tensors from disk, searching recursively."""
    def __init__(self, latent_dir, max_samples=None):
        self.latent_dir = latent_dir
        if not os.path.isdir(latent_dir):
            raise FileNotFoundError(f"Latent directory not found: {latent_dir}")

        # --- MODIFIED LINE ---
        # Use recursive=True or the **/ pattern to search subdirectories
        search_pattern = os.path.join(latent_dir, "**/*.pt")
        print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Searching for latents using pattern: {search_pattern}")
        self.latent_paths = sorted(glob.glob(search_pattern, recursive=True))
        # --- END MODIFIED LINE ---


        if not self.latent_paths:
            # Raise error specifically indicating no files found recursively
            raise FileNotFoundError(f"No latent files (.pt) found recursively under {latent_dir}.")


        # Print number found only on rank 0 to avoid clutter
        if dist.is_initialized() and dist.get_rank() == 0:
             print(f"Found {len(self.latent_paths)} latent files recursively.")

        if max_samples is not None:
            # Ensure sampling happens identically across ranks if needed, e.g., by setting seed
            # random.seed(42) # Example: Use a fixed seed for deterministic sampling across ranks
            if len(self.latent_paths) < max_samples:
                 print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Warning: Requested {max_samples} samples, but only found {len(self.latent_paths)}.")
            else:
                # Make sure sampling is consistent across processes if you sample
                # One way is to let rank 0 sample, then broadcast the list,
                # or use a fixed random seed before sampling on all ranks.
                # Simplest for now: sample independently on each rank (if dataset is large enough)
                # For truly identical sampling:
                # 1. Rank 0 samples indices or paths.
                # 2. Broadcast the list of paths/indices to other ranks.
                # Let's stick to independent sampling with a warning for now, assuming large dataset
                self.latent_paths = random.sample(self.latent_paths, max_samples)
                if dist.is_initialized() and dist.get_rank() == 0:
                    print(f"Sampled {max_samples} latents from {len(self.latent_paths)} found (sampled independently per rank).")


        # Rest of the __init__ remains the same...
        print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Initialized LatentDataset with {len(self.latent_paths)} paths.")
        try:
             first_latent = torch.load(self.latent_paths[0], map_location='cpu')
             self.dtype = first_latent.dtype
             if dist.is_initialized() and dist.get_rank() == 0: print(f"Detected latent dtype from files: {self.dtype}")
        except Exception as e:
             if dist.is_initialized() and dist.get_rank() == 0: print(f"Warning: Could not read first latent file to detect dtype: {e}. Assuming float.")
             self.dtype = torch.float32 # Fallback

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        # This part remains the same
        latent_path = self.latent_paths[idx]
        try:
            latent = torch.load(latent_path, map_location='cpu')
            return latent.float()
        except Exception as e:
            # Minimal logging in DDP
            # if dist.is_initialized() and dist.get_rank() == 0: print(f"ERROR loading {latent_path}")
            return None # Collate function needs to handle this


# --- Collate Function for Latent Dataset ---
def latent_collate_fn(batch):
    """Filters out None values from the batch and stacks valid tensors."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch failed
    return torch.stack(batch)

# --- Diffusion Model Utilities ---
def get_unet_model(args):
    """Creates the UNet model based on args."""
    # Parse tuple arguments from comma-separated strings
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

def get_vae_for_sampling(args):
    """Loads the VAE model specified in args, intended ONLY for rank 0 sampling."""
    # Reuse the VAE loading logic, simpler version for sampling
    if not args.vae_model_path or not os.path.exists(args.vae_model_path):
         print("Warning: VAE path for sampling not found or not specified. Using default TAESD.")
         model_name = "madebyollin/taesd" if args.vae_sd_version == 'v2.1' else "madebyollin/taesdxl"
         try:
             vae = AutoencoderTiny.from_pretrained(model_name, torch_dtype=torch.float32) # Use float32 for stability
             print(f"Loaded default VAE: {model_name}")
         except Exception as e:
             print(f"Error loading default VAE {model_name}: {e}. Sampling will fail.")
             return None
    else:
         print(f"Loading VAE for sampling from: {args.vae_model_path}")
         # Assume config matches the one used for training/precompute
         vae = AutoencoderTiny(latent_channels=args.unet_in_channels) # Basic structure
         try:
             vae.load_state_dict(torch.load(args.vae_model_path, map_location='cpu'))
             print("Successfully loaded VAE for sampling.")
         except Exception as e:
             print(f"Error loading VAE from {args.vae_model_path}: {e}. Sampling might fail.")
             # Optionally fall back to default VAE here?
             # return None
    if not hasattr(vae.config, 'scaling_factor'):
         vae.config.scaling_factor = 1.0 # TAESD default scaling
    return vae

# --- Sampling Function (Runs only on Rank 0) ---
@torch.no_grad()
def sample_latent_diffusion(unet_module, vae, scheduler, device, args, current_epoch, writer):
    """Generates and logs images from noise using the UNet module and VAE (Rank 0 only)."""
    unet_module.eval() # Use the unwrapped module
    if vae is None:
        print("VAE not available for sampling.")
        unet_module.train()
        return
    vae.to(device)
    vae.eval()

    num_samples = args.sampling_batch_size
    latent_shape = (num_samples, args.unet_in_channels, args.unet_sample_size, args.unet_sample_size)

    # Determine sampling dtype (use float32 for stability unless testing specific precision)
    sampling_dtype = torch.float32
    latents = torch.randn(latent_shape, device=device, dtype=sampling_dtype)

    dummy_encoder_hidden_states = torch.zeros(
        num_samples, 1, args.unet_cross_attention_dim,
        device=device, dtype=sampling_dtype
    )

    scheduler.set_timesteps(args.sampling_inference_steps)
    # tqdm for sampling steps (disable=False for rank 0)
    sampling_steps = tqdm(scheduler.timesteps, desc="Sampling", leave=False, disable=False)

    # No autocast during standard sampling loop for stability unless specifically desired
    for t in sampling_steps:
        latent_model_input = scheduler.scale_model_input(latents, t)
        # Pass tensors explicitly to the device
        t_tensor = torch.tensor([t] * num_samples, device=device).long() # Ensure t is tensor for model

        noise_pred = unet_module(
            sample=latent_model_input.to(device),
            timestep=t_tensor,
            encoder_hidden_states=dummy_encoder_hidden_states.to(device)
        ).sample

        latents = scheduler.step(noise_pred.to(device), t, latents).prev_sample # t should be scalar here

    # Decode
    # Use VAE scaling factor if available, else TAESD default
    scale_factor = getattr(vae.config, "scaling_factor", 1.0)
    latents_to_decode = latents / scale_factor
    decoded = vae.decode(latents_to_decode.to(device)).sample # Ensure latents are on VAE device

    images = (decoded.clamp(-1, 1) + 1) / 2
    images = images.cpu().float()

    grid = make_grid(images, nrow=max(1, num_samples // 2), normalize=True, value_range=(0, 1))
    writer.add_image('Generated_Samples', grid, current_epoch + 1) # Log samples to TensorBoard

    # Optionally save grid to disk
    save_dir = os.path.join(args.log_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{current_epoch+1}.png")
    save_image(grid, save_path)
    print(f"Saved sample grid to {save_path}")

    unet_module.train() # Set model back to train mode
    vae.cpu() # Move VAE back to CPU to save GPU memory


# --- Checkpoint Utilities (Modified for DDP) ---
def save_checkpoint(epoch, unet_module, optimizer, lr_scheduler, scaler, args):
    """Saves model, optimizer, scheduler, scaler state (Rank 0 only)."""
    checkpoint_name = (
        f"{args.checkpoint_filename_prefix}_"
        f"ep{epoch+1}_bs{args.batch_size_per_gpu}x{dist.get_world_size()}_"
        f"lr{args.lr}_acc{args.accumulation_steps}.pt"
    )
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'unet_state_dict': unet_module.state_dict(), # Save the unwrapped module's state
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'args': vars(args) # Save args for reproducibility
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")


def load_checkpoint(unet_module, optimizer, lr_scheduler, scaler, device, args):
    """Loads checkpoint state into model, optimizer, scheduler, scaler (All Ranks)."""
    # Construct expected checkpoint name pattern based on key parameters
    # This might need refinement if you change naming conventions significantly
    # Example: find latest matching prefix
    prefix = args.checkpoint_filename_prefix
    search_pattern = os.path.join(args.checkpoint_dir, f"{prefix}_ep*_bs*.pt")
    checkpoints = sorted(glob.glob(search_pattern), key=os.path.getmtime, reverse=True)

    start_epoch = 0
    if checkpoints:
        checkpoint_path = checkpoints[0] # Load the most recent one
        print(f"Rank {dist.get_rank()}: Loading checkpoint from {checkpoint_path}")
        try:
            # Load to the specific GPU for this rank
            map_location = {'cuda:0': f'cuda:{device.index}'} # Map rank 0's save to current rank's device
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

            # Load state dict into the unwrapped model module
            unet_module.load_state_dict(checkpoint['unet_state_dict'])
            print(f"Rank {dist.get_rank()}: Loaded UNet state dict.")

            if 'optimizer_state_dict' in checkpoint:
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 print(f"Rank {dist.get_rank()}: Loaded Optimizer state dict.")
            else: print(f"Rank {dist.get_rank()}: Optimizer state not found in checkpoint.")

            if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                 lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                 print(f"Rank {dist.get_rank()}: Loaded LR Scheduler state dict.")
            elif lr_scheduler: print(f"Rank {dist.get_rank()}: LR Scheduler state not found.")

            if 'scaler_state_dict' in checkpoint:
                 scaler.load_state_dict(checkpoint['scaler_state_dict'])
                 print(f"Rank {dist.get_rank()}: Loaded GradScaler state dict.")
            else: print(f"Rank {dist.get_rank()}: GradScaler state not found.")


            start_epoch = checkpoint.get('epoch', -1) + 1
            print(f"Rank {dist.get_rank()}: Resuming training from epoch {start_epoch}")
            # args_saved = checkpoint.get('args', {})
            # TODO: Compare saved args with current args if needed

        except Exception as e:
            print(f"Rank {dist.get_rank()}: ERROR loading checkpoint {checkpoint_path}: {e}")
            print(f"Rank {dist.get_rank()}: Starting training from scratch.")
            start_epoch = 0

    else:
        print(f"Rank {dist.get_rank()}: No checkpoint found matching pattern '{search_pattern}'. Training from scratch.")

    return start_epoch


# --- Training Function (Modified for DDP, bf16/fp16) ---
def train_diffusion_model(rank, world_size, local_rank, args):
    """Main DDP training loop."""
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (rank == 0)

    # --- Mixed Precision Setup ---
    amp_enabled = (args.mixed_precision in ["bf16", "fp16"])
    if amp_enabled:
        if args.mixed_precision == "bf16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
                print(f"Rank {rank}: Using bfloat16 mixed precision.")
            else:
                print(f"Rank {rank}: bf16 requested but not supported! Falling back to fp16.")
                amp_dtype = torch.float16
                if not (torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 7):
                     print(f"Rank {rank}: Warning: fp16 may have limited support on this GPU capability.")
        else: # fp16
            amp_dtype = torch.float16
            print(f"Rank {rank}: Using float16 mixed precision.")
            if not (torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 7):
                 print(f"Rank {rank}: Warning: fp16 may have limited support on this GPU capability.")
    else:
        amp_dtype = torch.float32
        print(f"Rank {rank}: Mixed precision disabled. Using float32.")

    # GradScaler is generally recommended with autocast, even for bf16
    scaler = GradScaler(enabled=(amp_enabled)) # Enable based on AMP type

    # --- TensorBoard (Rank 0 Only) ---
    writer = None
    if is_main_process:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log_dir = os.path.join(args.log_dir, f"run_{current_time}")
        os.makedirs(run_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_log_dir)
        print(f"TensorBoard logs will be saved to: {run_log_dir}")
        # Save config args
        with open(os.path.join(run_log_dir, "config_args.txt"), "w") as f:
            for k, v in sorted(vars(args).items()):
                f.write(f"{k}: {v}\n")


    # --- Dataset and DataLoader ---
    dataset = LatentDataset(args.latent_dir, max_samples=args.max_samples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=latent_collate_fn # Use the filtering collate function
    )

    # --- Models, Scheduler, Optimizer ---
    unet = get_unet_model(args)
    # Load VAE only on Rank 0 for sampling purposes
    vae = get_vae_for_sampling(args) if is_main_process else None

    # Move UNet to the correct device *before* DDP wrapping and optimizer init
    unet.to(device)

    scheduler = DDPMScheduler(num_train_timesteps=args.scheduler_train_timesteps)
    optimizer = optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-3)

    # Calculate total steps for LR scheduler
    num_batches_per_epoch = len(dataloader) # Batches per GPU per epoch
    # Num update steps PER GPU per epoch
    num_update_steps_per_epoch = math.ceil(num_batches_per_epoch / args.accumulation_steps)
    total_train_steps = args.epochs * num_update_steps_per_epoch

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_train_steps, eta_min=1e-6
    )
    if is_main_process:
        print(f"LR Scheduler: Cosine Annealing with T_max={total_train_steps} steps.")

    # --- Load Checkpoint (All Ranks) ---
    # Load state into the base model (unet) before wrapping
    start_epoch = load_checkpoint(unet, optimizer, lr_scheduler, scaler, device, args)

    # --- Wrap Model with DDP ---
    # find_unused_parameters can be helpful for debugging certain DDP issues
    unet = DDP(unet, device_ids=[local_rank])#, find_unused_parameters=True)

    # --- Training Loop ---
    global_step = start_epoch * num_update_steps_per_epoch
    total_start_time = time.time()

    if is_main_process:
         print(f"Starting UNet DDP training from epoch {start_epoch+1}/{args.epochs}...")
         print(f"  World Size: {world_size}")
         print(f"  Batch Size Per GPU: {args.batch_size_per_gpu}")
         print(f"  Gradient Accumulation Steps: {args.accumulation_steps}")
         print(f"  Effective Total Batch Size: {args.batch_size_per_gpu * world_size * args.accumulation_steps}")
         print(f"  Mixed Precision: {args.mixed_precision} (dtype: {amp_dtype}, AMP: {amp_enabled})")
         print(f"  Number of Latent Samples: {len(dataset)}")
         print(f"  Steps per Epoch (per GPU): ~{num_update_steps_per_epoch}")

    unet.train() # Ensure train mode

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch) # Ensure proper shuffling with DDP
        epoch_loss = 0.0
        epoch_start_time = time.time()

        # Progress bar only on Rank 0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not is_main_process,
            leave=False
        )

        optimizer.zero_grad() # Zero gradients at the start of accumulation cycle

        for batch_idx, latents in enumerate(progress_bar):
            if latents is None: # Skip if collate_fn returned None
                 if is_main_process: print(f"Warning: Skipping None batch at index {batch_idx}")
                 continue

            # Data is already on 'device' due to pin_memory and DDP handling
            # Ensure latents are float32 for model input standard practice
            latents = latents.to(device, dtype=torch.float32)

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            b = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (b,), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Dummy condition
            unet_dtype = next(unet.parameters()).dtype # Get current dtype (can change with autocast)
            dummy_encoder_hidden_states = torch.zeros(
                b, 1, args.unet_cross_attention_dim, device=device, dtype=unet_dtype
            )

            # Predict noise with AMP
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                noisy_latents_input = scheduler.scale_model_input(noisy_latents, timesteps)
                noise_pred = unet( # DDP forward pass
                    sample=noisy_latents_input,
                    timestep=timesteps,
                    encoder_hidden_states=dummy_encoder_hidden_states
                ).sample
                # Calculate loss in float32 for stability
                loss = F.mse_loss(noise_pred.float(), noise.float())
                loss = loss / args.accumulation_steps # Scale loss for accumulation

            # Backpropagate (DDP handles gradient averaging)
            scaler.scale(loss).backward()

            # Accumulate unscaled loss for logging (use item() to avoid tensor memory leak)
            # Note: This loss is local to the current rank's batch
            batch_loss_unscaled = loss.item() * args.accumulation_steps
            epoch_loss += batch_loss_unscaled

            # Optimizer Step
            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Optional: Gradient clipping (before optimizer step)
                # scaler.unscale_(optimizer) # Unscale first
                # torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # Zero gradients AFTER step and update
                lr_scheduler.step() # Step LR scheduler after optimizer step

                global_step += 1

                # Logging (Rank 0 Only)
                if is_main_process and global_step % args.log_interval == 0:
                    # Gather loss from all ranks for more accurate logging (optional)
                    # loss_tensor = torch.tensor([batch_loss_unscaled], device=device)
                    # dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    # avg_loss_across_gpus = loss_tensor.item()
                    # writer.add_scalar('Train/Batch_Loss_Avg', avg_loss_across_gpus, global_step)

                    # Log local rank 0 loss for simplicity
                    writer.add_scalar('Train/Batch_Loss_Rank0', batch_loss_unscaled, global_step)
                    writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                    # Log scaler state if needed
                    # writer.add_scalar('Train/GradScaler_Scale', scaler.get_scale(), global_step)

            if is_main_process:
                progress_bar.set_postfix(loss=f"{batch_loss_unscaled:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}", step=global_step)

        # --- End of Epoch ---
        # Average loss calculation needs care in DDP
        # epoch_loss is sum of local losses on rank 0. Divide by number of batches processed on rank 0.
        avg_epoch_loss_rank0 = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0

        # Optional: Gather average loss across all ranks
        avg_loss_tensor = torch.tensor([avg_epoch_loss_rank0], device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_epoch_loss_all = avg_loss_tensor.item()


        epoch_duration = time.time() - epoch_start_time
        total_elapsed = time.time() - total_start_time

        if is_main_process:
            print(f"\nEpoch {epoch+1} completed. Avg Loss (Rank 0): {avg_epoch_loss_rank0:.6f}, Avg Loss (All Ranks): {avg_epoch_loss_all:.6f}")
            print(f"Time: {str(datetime.timedelta(seconds=int(epoch_duration)))}, Total: {str(datetime.timedelta(seconds=int(total_elapsed)))}")
            writer.add_scalar('Train/Epoch_Avg_Loss_Rank0', avg_epoch_loss_rank0, epoch + 1)
            writer.add_scalar('Train/Epoch_Avg_Loss_All', avg_epoch_loss_all, epoch + 1)
            writer.add_scalar('Train/Epoch_Duration_sec', epoch_duration, epoch + 1)

            # Sampling (Rank 0 Only)
            if (epoch + 1) % args.sample_interval == 0:
                print("Generating samples...")
                sample_latent_diffusion(unet.module, vae, scheduler, device, args, epoch, writer)

            # Save Checkpoint (Rank 0 Only)
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(epoch, unet.module, optimizer, lr_scheduler, scaler, args)

        # Barrier to ensure all processes finish epoch before starting next one or finishing
        dist.barrier()

    # --- End of Training ---
    if is_main_process:
        print(f"UNet training finished! Total time: {str(datetime.timedelta(seconds=int(time.time() - total_start_time)))}")
        if writer:
            writer.close()
    dist.barrier()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Diffusion Model Training Script")

    # Add arguments based on TrainingConfig.get_script_args() generation logic
    parser.add_argument('--latent_dir', type=str, required=True, help="Directory containing pre-computed latents (.pt files)")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument('--log_dir', type=str, required=True, help="Directory for TensorBoard logs")
    parser.add_argument('--vae_model_path', type=str, required=True, help="Path to the trained VAE model (for sampling on rank 0)")
    parser.add_argument('--vae_sd_version', type=str, default='v2.1', choices=['v2.1', 'sdxl'], help="Version of the VAE used (e.g., 'v2.1', 'sdxl')")

    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--batch_size_per_gpu', type=int, default=32, help="Batch size for each GPU")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--accumulation_steps', type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument('--mixed_precision', type=str, default='bf16', choices=['bf16', 'fp16', 'no'], help="Mixed precision type ('bf16', 'fp16', 'no')")

    # UNet specific arguments
    parser.add_argument('--unet_sample_size', type=int, default=64, help="Input patch size for UNet (latent size)")
    parser.add_argument('--unet_in_channels', type=int, default=4, help="Input channels for UNet (usually VAE latent channels)")
    parser.add_argument('--unet_out_channels', type=int, default=4, help="Output channels for UNet (usually VAE latent channels)")
    parser.add_argument('--unet_block_out_channels', type=str, default='320,640,1280,1280', help="Comma-separated list of UNet block output channels")
    parser.add_argument('--unet_down_block_types', type=str, default='CrossAttnDownBlock2D,CrossAttnDownBlock2D,CrossAttnDownBlock2D,DownBlock2D', help="Comma-separated list of UNet down block types")
    parser.add_argument('--unet_up_block_types', type=str, default='UpBlock2D,CrossAttnUpBlock2D,CrossAttnUpBlock2D,CrossAttnUpBlock2D', help="Comma-separated list of UNet up block types")
    parser.add_argument('--unet_cross_attention_dim', type=int, default=768, help="Dimension of UNet cross-attention")

    # Scheduler and Sampling arguments
    parser.add_argument('--scheduler_train_timesteps', type=int, default=1000, help="Number of timesteps for the DDPMScheduler during training")
    parser.add_argument('--sampling_inference_steps', type=int, default=50, help="Number of inference steps for sampling")
    parser.add_argument('--sampling_batch_size', type=int, default=6, help="Number of samples to generate during training visualization (rank 0)")

    # Data loading and logging/saving arguments
    parser.add_argument('--num_workers', type=int, default=8, help="Number of DataLoader workers per GPU")
    parser.add_argument('--checkpoint_filename_prefix', type=str, default="unet_ddp", help="Prefix for saved checkpoint filenames")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of latent samples to use from the dataset (optional)")
    parser.add_argument('--log_interval', type=int, default=10, help="Log training loss every N global steps (rank 0)")
    parser.add_argument('--sample_interval', type=int, default=1, help="Generate samples every N epochs (rank 0)")
    parser.add_argument('--save_interval', type=int, default=1, help="Save checkpoint every N epochs (rank 0)")

    args = parser.parse_args()

    # --- Initialize DDP ---
    rank, world_size, local_rank = setup_ddp()

    try:
        # --- Start Training ---
        train_diffusion_model(rank, world_size, local_rank, args)
    except Exception as e:
        print(f"Rank {rank}: Encountered exception during training: {e}", file=sys.stderr)
        # Optional: Re-raise the exception if needed for better traceback in logs
        # raise e
    finally:
        # --- Cleanup DDP ---
        cleanup_ddp()