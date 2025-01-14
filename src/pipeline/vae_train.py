"""
train_vae.py

一个用于训练基于 diffusers 的自动编码器 (VAE) 的脚本示例。
在此版本中，AutoencoderKL 模型需要在外部定义，并作为参数传入 train_vae()。

Requirements:
    - torch
    - torchvision
    - diffusers
    - numpy
    - Pillow
    - matplotlib
    - tqdm
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 如果在非 notebook 环境下使用，推荐使用 from tqdm import tqdm
# 这里演示 notebook 的做法，如果要在脚本或终端中运行，替换为 from tqdm import tqdm
from tqdm.notebook import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


class PositivePatchDataset(Dataset):
    """
    将存有 [N, 1, 128, 128] 形状图像补丁的 Tensor 打包成可用于 DataLoader 的数据集。
    在 __getitem__ 中采用 PIL + transforms 的形式进行数据增强。
    """

    def __init__(self, patches: torch.Tensor):
        """
        Args:
            patches (torch.Tensor): Tensor of shape [N, 1, 128, 128]
                                    通常在 CPU 上即可 (避免过早占用 GPU)
        """
        super().__init__()
        self.patches = patches
        # 定义数据增强 transforms
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            # 注意：如果 patch 已经是 torch.Tensor，这里也可以选择只做随机变换。但为了复用 torchvision.transforms 的便利，这里先转 PIL，再转回 Tensor。
            T.ToTensor(),  # 最终保证输出是 [C, H, W]，值域 [0,1]
        ])

    def __len__(self) -> int:
        return self.patches.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            torch.Tensor of shape [1, 128, 128] in range [0,1].
        """
        # patch: [1, 128, 128]
        patch = self.patches[idx]

        # 转换为 [0,255] 的 uint8，方便转成 PIL
        patch_np = (patch.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        # 由 numpy 转 PIL (灰度图 'L')
        patch_pil = Image.fromarray(patch_np, mode='L')

        # 经过 transform
        patch_transformed = self.transform(patch_pil)
        return patch_transformed


def visualize_reconstructions(
    model, 
    dataloader: DataLoader, 
    device: torch.device, 
    epoch: int, 
    num_images: int = 5
):
    """
    可视化原图与重建图对比。

    Args:
        model: 训练中的 VAE 模型 (如 AutoencoderKL)。
        dataloader (DataLoader): 用于抽取一批数据进行可视化。
        device (torch.device): 设备。
        epoch (int): 当前的 epoch，用于在标题中标注。
        num_images (int): 可视化多少张图片的 (原图, 重建图)。
    """
    model.eval()
    with torch.no_grad():
        # 从数据集中取一批
        batch_data = next(iter(dataloader))
        batch_data = batch_data.to(device)

        # 编码
        latent_dist = model.encode(batch_data).latent_dist  # 这里得到的是分布
        # 在分布上采样，并进行解码
        recon_data = model.decode(latent_dist.sample()).sample

        # 移动到 CPU 并转成 numpy
        batch_data = batch_data.cpu().numpy()
        recon_data = recon_data.cpu().numpy()

    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # 原图
        plt.subplot(2, num_images, i + 1)
        plt.imshow(batch_data[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # 重建图
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(recon_data[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.suptitle(f"Reconstruction at Epoch {epoch}")
    plt.tight_layout()
    plt.show()


def train_vae(
    model,
    graph_data_dict: dict,
    positive_nodes_dict: dict,
    save_dir: str = 'vae_checkpoints',
    checkpoint_prefix: str = 'vae',  # 新增的文件名前缀参数
    num_epochs: int = 20,
    batch_size: int = 64,
    warmup_epochs: int = 10,
    lr: float = 1e-4,
    save_interval: int = 5,
    visualize_frequency: int = 5,
    device: torch.device = None,
    show_images: bool = True,
    save_checkpoint: bool = True,
    loss_fn: nn.Module = None
):
    """
    训练一个 VAE 模型，流程包括：
        1) 从 graph_data_dict 及 positive_nodes_dict 中提取补丁
        2) 构建 PositivePatchDataset 并划分 train/val
        3) 使用指定模型 (model) 进行训练
        4) 可选输出训练和验证的曲线，可视化重建结果，保存模型

    Args:
        model: 你自行定义好的 VAE 模型 (例如 AutoencoderKL 对象)。
        graph_data_dict (dict): {key -> data}, data 里包含 data.patches: [N, H, W, C]
        positive_nodes_dict (dict): {key -> positive_nodes}, 表示哪些节点是 positive
        save_dir (str): 模型保存目录
        checkpoint_prefix (str): 模型保存文件名前缀
        num_epochs (int): 训练轮数
        batch_size (int): 每批数据大小
        warmup_epochs (int): KL loss 的 warm-up 轮数
        lr (float): 学习率
        save_interval (int): 每多少轮保存一次模型
        visualize_frequency (int): 每多少轮可视化重建结果
        device (torch.device): 训练所用设备，默认为 CUDA (若可用)
        show_images (bool): 是否显示可视化重建结果
        save_checkpoint (bool): 是否在 save_interval 处保存模型
        loss_fn (nn.Module): 自定义的重建损失函数，若为 None 则默认使用 BCEWithLogitsLoss(reduction='mean')

    Returns:
        model: 训练完成的 VAE 模型
        (train_losses, val_losses): 训练和验证的 loss 曲线
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型移动到指定设备
    model.to(device)

    os.makedirs(save_dir, exist_ok=True)

    # ------------ 1) 收集正样本补丁 ------------
    positive_patches = []
    for key, data in graph_data_dict.items():
        # 取出 positive nodes 的索引
        positive_nodes = positive_nodes_dict[key].cpu()  # 确保在 CPU
        # data.patches 的形状一般是 [N, H, W, C]，这里转 [N, C, H, W]
        # 先索引得到 [num_positive, H, W, C]，再 permute 得到 [num_positive, C, H, W]
        positive_patches_sample = data.patches[positive_nodes].permute(0, 3, 1, 2)
        positive_patches.append(positive_patches_sample)

    # 拼接成一个 [total_positive, 1, 128, 128]
    positive_patches_tensor = torch.cat(positive_patches, dim=0)

    # ------------ 2) 构建数据集，并划分 train/val ------------
    positive_dataset = PositivePatchDataset(positive_patches_tensor)

    train_size = int(0.8 * len(positive_dataset))
    val_size = len(positive_dataset) - train_size
    train_dataset, val_dataset = random_split(positive_dataset, [train_size, val_size])

    num_workers = 0  # 如果在 Windows 或交互式环境（notebook）下，常常设为 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 组装个字典管理
    plaque_loader = {'train': train_loader, 'val': val_loader}

    # 如果没有指定自定义损失函数，则使用默认 BCEWithLogitsLoss
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    # 用于计算 KL 散度
    def kl_divergence_loss(mu, logvar):
        # KL 散度: D_KL( q(z|x) || p(z) ) = -0.5 * (1 + logvar - mu^2 - exp(logvar))
        # 这里对 batch 取平均
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # (可选) 学习率调度
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    val_losses = []

    # ------------ 3) 训练循环 ------------
    for epoch in range(1, num_epochs + 1):
        # ---------- Training ----------
        model.train()
        epoch_train_loss = 0.0
        train_bar = tqdm(
            plaque_loader['train'],
            desc=f"Epoch {epoch}/{num_epochs} [Train]",
            leave=False
        )
        for batch_idx, data in enumerate(train_bar):
            data = data.to(device)
            optimizer.zero_grad()

            # 前向：Encode -> Decode
            latent_dist = model.encode(data).latent_dist
            recon_data = model.decode(latent_dist.sample()).sample

            # 均值和对数方差
            mu, logvar = latent_dist.mean, latent_dist.logvar

            # 计算损失
            recon_loss = loss_fn(recon_data, data)
            kl_loss = kl_divergence_loss(mu, logvar)

            # KL warm-up
            annealing_factor = min(1.0, epoch / warmup_epochs)
            loss = recon_loss + annealing_factor * kl_loss

            # 反向传播
            loss.backward()
            # (可选)梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            train_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Recon': f"{recon_loss.item():.4f}",
                'KL': f"{kl_loss.item():.4f}"
            })

        avg_train_loss = epoch_train_loss / len(plaque_loader['train'])
        train_losses.append(avg_train_loss)

        # ---------- Validation ----------
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(
                plaque_loader['val'],
                desc=f"Epoch {epoch}/{num_epochs} [Val]",
                leave=False
            )
            for data in val_bar:
                data = data.to(device)
                latent_dist = model.encode(data).latent_dist
                recon_data = model.decode(latent_dist.sample()).sample

                mu, logvar = latent_dist.mean, latent_dist.logvar

                recon_loss = loss_fn(recon_data, data)
                kl_loss = kl_divergence_loss(mu, logvar)

                # 验证阶段仍沿用相同的 annealing_factor
                loss = recon_loss + annealing_factor * kl_loss

                epoch_val_loss += loss.item()
                val_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Recon': f"{recon_loss.item():.4f}",
                    'KL': f"{kl_loss.item():.4f}"
                })

        avg_val_loss = epoch_val_loss / len(plaque_loader['val'])
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        # (可选) 学习率调度
        # scheduler.step()

 # -------- 保存模型权重 --------
        if save_checkpoint and ((epoch % save_interval == 0) or (epoch == num_epochs)):
            checkpoint_path = os.path.join(save_dir, f"{checkpoint_prefix}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint at: {checkpoint_path}")
            
        # 周期性可视化
        if show_images and ((epoch % visualize_frequency == 0) or (epoch == num_epochs)):
            visualize_reconstructions(model, plaque_loader['val'], device, epoch, num_images=5)

    # ---------- Loss 曲线 ----------
    if show_images:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    return model, (train_losses, val_losses)
