import os
import sys
import random
import glob
import datetime
import time
import subprocess
import shlex
import warnings
from pathlib import Path
os.chdir("/home1/jijh/diffusion_project/ADiffusion")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader # Use PyG's DataLoader
from torch_geometric.nn import GPSConv, GATConv, global_add_pool
from torch_geometric.utils.convert import from_scipy_sparse_matrix

# Assume hest_loading is in the path or installed
from src.pipeline.hest_loading import HESTDataset, HESTSample

# Filter annoying warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", message="Observation names are not unique.")
warnings.filterwarnings("ignore", message="Variable names are not unique.")

# Cell 2: Configuration Class
class TrainingConfig:
    # --- Hardware & Precision ---
    GPU_IDS = [0,1,2] # <-- SPECIFY THE DESIRED GPU INDICES HERE
    PRIMARY_GPU_ID = GPU_IDS[0] if GPU_IDS else 0
    NUM_GPUS = len(GPU_IDS) if torch.cuda.is_available() and GPU_IDS else 0 # Number of GPUs to USE
    PRIMARY_DEVICE_NAME = f"cuda:{PRIMARY_GPU_ID}" if NUM_GPUS > 0 else "cpu"
    MIXED_PRECISION_TYPE = "bf16" # "bf16", "fp16", or "no"
    DDP_MASTER_PORT = 29502 # Choose an unused port


    # --- Data Paths ---
    HEST_DATA_DIR = "/cwStorage/nodecw_group/jijh/hest_1k"
    # Input VAE Latents (from previous step)
    LATENT_DIR = "/cwStorage/nodecw_group/jijh/hest_output_latents_bf16"
    # Output Directory for Processed Graphs
    GRAPH_DATA_DIR = "/cwStorage/nodecw_group/jijh/hest_graph_data_pca50_knn6"
    # Input VAE Model
    VAE_MODEL_PATH = "/cwStorage/nodecw_group/jijh/model_path/finetuned_taesd_v21_notebook_apr2.pt"
    # Input Pre-trained UNet (optional, for fine-tuning)
    PRETRAINED_UNET_PATH = "/cwStorage/nodecw_group/jijh/model_path/unet_ddp_bf16_ep15_bs32x3_lr0.0001_acc4.pt" # Example path from previous run


    # --- Preprocessing ---
    PCA_N_COMPS = 50
    SPATIAL_N_NEIGHBORS = 6 # KNN for graph construction
    # Limit samples for faster testing?
    PREPROCESS_MAX_SAMPLES = None # Set to a number (e.g., 10) for testing

    # --- VAE ---
    VAE_SD_VERSION = 'v2.1'
    VAE_LATENT_CHANNELS = 4 # Should match precomputed latents

    # --- Graph Conditioner (GPSConv based) ---
    CONDITIONER_INPUT_DIM = PCA_N_COMPS
    CONDITIONER_HIDDEN_DIM = 256 # Internal dimension of GPSConv
    CONDITIONER_OUTPUT_DIM = 768 # MUST match UNet cross_attention_dim
    CONDITIONER_N_LAYERS = 4    # Number of GPSConv layers
    CONDITIONER_N_HEADS = 4     # Heads for multi-head attention in GPSConv
    CONDITIONER_ATTN_DROPOUT = 0.1

    # --- Diffusion (UNet) --- Parameters must match the pre-trained one if loading
    UNET_SAMPLE_SIZE = 64  # Latent spatial dimensions
    UNET_IN_CHANNELS = VAE_LATENT_CHANNELS
    UNET_OUT_CHANNELS = VAE_LATENT_CHANNELS
    UNET_BLOCK_OUT_CHANNELS = (320, 640, 1280, 1280)
    UNET_DOWN_BLOCK_TYPES = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D')
    UNET_UP_BLOCK_TYPES = ('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D')
    UNET_CROSS_ATTENTION_DIM = CONDITIONER_OUTPUT_DIM # Ensure match

    # --- Training Script Config ---
    DIFFUSION_BATCH_SIZE_PER_GPU = 4 # Adjust based on GPU memory
    DIFFUSION_NUM_EPOCHS = 20
    DIFFUSION_LEARNING_RATE = 5e-5 # May need tuning
    ACCUMULATION_STEPS = 4
    NUM_WORKERS = 16 # DataLoader workers
    SCHEDULER_TRAIN_TIMESTEPS = 1000
    SAMPLING_INFERENCE_STEPS = 50 # For visualization during training
    # SAMPLING_BATCH_SIZE = 16 # <-- NO LONGER NEEDED HERE, fixed in script

    # --- Logging & Saving (Script) ---
    CHECKPOINT_DIR = "/cwStorage/nodecw_group/jijh/model_path/conditioned_diffusion_v1"
    LOG_DIR = "/cwStorage/nodecw_group/jijh/training_log/conditioned_diffusion_v1"
    CHECKPOINT_FILENAME_PREFIX = "cond_unet_gps"
    TRAIN_SCRIPT_PATH = "/home1/jijh/diffusion_project/ADiffusion/src/pipeline/train_condition_diffusion_ddp.py" # Path to the DDP script
    # NEW: Define step-based sampling interval here
    SAMPLE_INTERVAL_STEPS = 200
    SAVE_INTERVAL_EPOCHS = 1 # Save every epoch by default now

    @classmethod
    def get_script_args(cls):
        """Generates CLI arguments for the DDP training script."""
        # This method remains unchanged, it correctly uses config values
        args = [
            f"--graph_data_dir={cls.GRAPH_DATA_DIR}",
            f"--latent_dir={cls.LATENT_DIR}",
            f"--checkpoint_dir={cls.CHECKPOINT_DIR}",
            f"--log_dir={cls.LOG_DIR}",
            f"--vae_model_path={cls.VAE_MODEL_PATH}",
            f"--vae_sd_version={cls.VAE_SD_VERSION}",
            f"--pretrained_unet_path={cls.PRETRAINED_UNET_PATH}" if cls.PRETRAINED_UNET_PATH else "--pretrained_unet_path=None",

            f"--epochs={cls.DIFFUSION_NUM_EPOCHS}",
            f"--batch_size_per_gpu={cls.DIFFUSION_BATCH_SIZE_PER_GPU}",
            f"--lr={cls.DIFFUSION_LEARNING_RATE}",
            f"--accumulation_steps={cls.ACCUMULATION_STEPS}",
            f"--mixed_precision={cls.MIXED_PRECISION_TYPE}",
            f"--num_workers={cls.NUM_WORKERS}",

            f"--pca_n_comps={cls.PCA_N_COMPS}", # Pass PCA info
            f"--conditioner_input_dim={cls.CONDITIONER_INPUT_DIM}",
            f"--conditioner_hidden_dim={cls.CONDITIONER_HIDDEN_DIM}",
            f"--conditioner_output_dim={cls.CONDITIONER_OUTPUT_DIM}",
            f"--conditioner_n_layers={cls.CONDITIONER_N_LAYERS}",
            f"--conditioner_n_heads={cls.CONDITIONER_N_HEADS}",
            f"--conditioner_attn_dropout={cls.CONDITIONER_ATTN_DROPOUT}",

            f"--unet_sample_size={cls.UNET_SAMPLE_SIZE}",
            f"--unet_in_channels={cls.UNET_IN_CHANNELS}",
            f"--unet_out_channels={cls.UNET_OUT_CHANNELS}",
            f"--unet_block_out_channels={','.join(map(str, cls.UNET_BLOCK_OUT_CHANNELS))}",
            f"--unet_down_block_types={','.join(cls.UNET_DOWN_BLOCK_TYPES)}",
            f"--unet_up_block_types={','.join(cls.UNET_UP_BLOCK_TYPES)}",
            f"--unet_cross_attention_dim={cls.UNET_CROSS_ATTENTION_DIM}",

            f"--scheduler_train_timesteps={cls.SCHEDULER_TRAIN_TIMESTEPS}",
            f"--sampling_inference_steps={cls.SAMPLING_INFERENCE_STEPS}",
            f"--sample_interval_steps={cls.SAMPLE_INTERVAL_STEPS}",
            f"--save_interval={cls.SAVE_INTERVAL_EPOCHS}",
            f"--checkpoint_filename_prefix={cls.CHECKPOINT_FILENAME_PREFIX}",
            f"--log_interval=20",
        ]
        return args

# --- Instantiate config (run this cell again!) ---
config = TrainingConfig()
os.makedirs(config.GRAPH_DATA_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)
# --- End of Cell 2 ---
# Cell 2: Configuration Class
class TrainingConfig:
    # --- Hardware & Precision ---
    GPU_IDS = [0, 2] # <-- SPECIFY THE DESIRED GPU INDICES HERE
    PRIMARY_GPU_ID = GPU_IDS[0] if GPU_IDS else 0
    NUM_GPUS = len(GPU_IDS) if torch.cuda.is_available() and GPU_IDS else 0 # Number of GPUs to USE
    PRIMARY_DEVICE_NAME = f"cuda:{PRIMARY_GPU_ID}" if NUM_GPUS > 0 else "cpu"
    MIXED_PRECISION_TYPE = "bf16" # "bf16", "fp16", or "no"
    DDP_MASTER_PORT = 29502 # Choose an unused port


    # --- Data Paths ---
    HEST_DATA_DIR = "/cwStorage/nodecw_group/jijh/hest_1k"
    # Input VAE Latents (from previous step)
    LATENT_DIR = "/cwStorage/nodecw_group/jijh/hest_output_latents_bf16"
    # Output Directory for Processed Graphs
    GRAPH_DATA_DIR = "/cwStorage/nodecw_group/jijh/hest_graph_data_pca50_knn6"
    # Input VAE Model
    VAE_MODEL_PATH = "/cwStorage/nodecw_group/jijh/model_path/finetuned_taesd_v21_notebook_apr2.pt"
    # Input Pre-trained UNet (optional, for fine-tuning)
    PRETRAINED_UNET_PATH = "/cwStorage/nodecw_group/jijh/model_path/unet_ddp_bf16_ep15_bs32x3_lr0.0001_acc4.pt" # Example path from previous run


    # --- Preprocessing ---
    PCA_N_COMPS = 50
    SPATIAL_N_NEIGHBORS = 6 # KNN for graph construction
    # Limit samples for faster testing?
    PREPROCESS_MAX_SAMPLES = None # Set to a number (e.g., 10) for testing

    # --- VAE ---
    VAE_SD_VERSION = 'v2.1'
    VAE_LATENT_CHANNELS = 4 # Should match precomputed latents

    # --- Graph Conditioner (GPSConv based) ---
    CONDITIONER_INPUT_DIM = PCA_N_COMPS
    CONDITIONER_HIDDEN_DIM = 256 # Internal dimension of GPSConv
    CONDITIONER_OUTPUT_DIM = 768 # MUST match UNet cross_attention_dim
    CONDITIONER_N_LAYERS = 4    # Number of GPSConv layers
    CONDITIONER_N_HEADS = 4     # Heads for multi-head attention in GPSConv
    CONDITIONER_ATTN_DROPOUT = 0.1

    # --- Diffusion (UNet) --- Parameters must match the pre-trained one if loading
    UNET_SAMPLE_SIZE = 64  # Latent spatial dimensions
    UNET_IN_CHANNELS = VAE_LATENT_CHANNELS
    UNET_OUT_CHANNELS = VAE_LATENT_CHANNELS
    UNET_BLOCK_OUT_CHANNELS = (320, 640, 1280, 1280)
    UNET_DOWN_BLOCK_TYPES = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D')
    UNET_UP_BLOCK_TYPES = ('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D')
    UNET_CROSS_ATTENTION_DIM = CONDITIONER_OUTPUT_DIM # Ensure match

    # --- Training Script Config ---
    DIFFUSION_BATCH_SIZE_PER_GPU = 4 # Adjust based on GPU memory
    DIFFUSION_NUM_EPOCHS = 20
    DIFFUSION_LEARNING_RATE = 5e-5 # May need tuning
    ACCUMULATION_STEPS = 4
    NUM_WORKERS = 16 # DataLoader workers
    SCHEDULER_TRAIN_TIMESTEPS = 1000
    SAMPLING_INFERENCE_STEPS = 50 # For visualization during training
    # SAMPLING_BATCH_SIZE = 16 # <-- NO LONGER NEEDED HERE, fixed in script

    # --- Logging & Saving (Script) ---
    CHECKPOINT_DIR = "/cwStorage/nodecw_group/jijh/model_path/conditioned_diffusion_v1"
    LOG_DIR = "/cwStorage/nodecw_group/jijh/training_log/conditioned_diffusion_v1"
    CHECKPOINT_FILENAME_PREFIX = "cond_unet_gps"
    TRAIN_SCRIPT_PATH = "/home1/jijh/diffusion_project/ADiffusion/src/pipeline/train_condition_diffusion_ddp.py" # Path to the DDP script
    # NEW: Define step-based sampling interval here
    SAMPLE_INTERVAL_STEPS = 200
    SAVE_INTERVAL_EPOCHS = 1 # Save every epoch by default now

    @classmethod
    def get_script_args(cls):
        """Generates CLI arguments for the DDP training script."""
        # This method remains unchanged, it correctly uses config values
        args = [
            f"--graph_data_dir={cls.GRAPH_DATA_DIR}",
            f"--latent_dir={cls.LATENT_DIR}",
            f"--checkpoint_dir={cls.CHECKPOINT_DIR}",
            f"--log_dir={cls.LOG_DIR}",
            f"--vae_model_path={cls.VAE_MODEL_PATH}",
            f"--vae_sd_version={cls.VAE_SD_VERSION}",
            f"--pretrained_unet_path={cls.PRETRAINED_UNET_PATH}" if cls.PRETRAINED_UNET_PATH else "--pretrained_unet_path=None",

            f"--epochs={cls.DIFFUSION_NUM_EPOCHS}",
            f"--batch_size_per_gpu={cls.DIFFUSION_BATCH_SIZE_PER_GPU}",
            f"--lr={cls.DIFFUSION_LEARNING_RATE}",
            f"--accumulation_steps={cls.ACCUMULATION_STEPS}",
            f"--mixed_precision={cls.MIXED_PRECISION_TYPE}",
            f"--num_workers={cls.NUM_WORKERS}",

            f"--pca_n_comps={cls.PCA_N_COMPS}", # Pass PCA info
            f"--conditioner_input_dim={cls.CONDITIONER_INPUT_DIM}",
            f"--conditioner_hidden_dim={cls.CONDITIONER_HIDDEN_DIM}",
            f"--conditioner_output_dim={cls.CONDITIONER_OUTPUT_DIM}",
            f"--conditioner_n_layers={cls.CONDITIONER_N_LAYERS}",
            f"--conditioner_n_heads={cls.CONDITIONER_N_HEADS}",
            f"--conditioner_attn_dropout={cls.CONDITIONER_ATTN_DROPOUT}",

            f"--unet_sample_size={cls.UNET_SAMPLE_SIZE}",
            f"--unet_in_channels={cls.UNET_IN_CHANNELS}",
            f"--unet_out_channels={cls.UNET_OUT_CHANNELS}",
            f"--unet_block_out_channels={','.join(map(str, cls.UNET_BLOCK_OUT_CHANNELS))}",
            f"--unet_down_block_types={','.join(cls.UNET_DOWN_BLOCK_TYPES)}",
            f"--unet_up_block_types={','.join(cls.UNET_UP_BLOCK_TYPES)}",
            f"--unet_cross_attention_dim={cls.UNET_CROSS_ATTENTION_DIM}",

            f"--scheduler_train_timesteps={cls.SCHEDULER_TRAIN_TIMESTEPS}",
            f"--sampling_inference_steps={cls.SAMPLING_INFERENCE_STEPS}",
            f"--sample_interval_steps={cls.SAMPLE_INTERVAL_STEPS}",
            f"--save_interval={cls.SAVE_INTERVAL_EPOCHS}",
            f"--checkpoint_filename_prefix={cls.CHECKPOINT_FILENAME_PREFIX}",
            f"--log_interval=20",
        ]
        return args

# --- Instantiate config (run this cell again!) ---
config = TrainingConfig()
os.makedirs(config.GRAPH_DATA_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)
# --- End of Cell 2 ---