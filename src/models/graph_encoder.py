# train_module.py

from typing import Optional, Tuple, List, Type
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.data import Data
from tqdm import tqdm
import random
import logging

# 配置 logging（可根据需要调整日志级别）
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def apply_mask(x: torch.Tensor, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    对输入特征 x 进行 BERT 风格的 masking 操作。

    参数:
        x (torch.Tensor): 原始节点特征，形状为 [num_nodes, num_features]。
        mask (torch.Tensor): 布尔型 mask，形状为 [num_nodes]；True 表示需要 mask。
        device (torch.device): 设备，用于保证生成的随机张量在同一设备上。

    返回:
        torch.Tensor: 被 mask 过后的特征张量（复制后的副本）。
    """
    x_masked = x.clone()
    indices = torch.where(mask)[0]
    for idx in indices:
        rnd = random.random()
        if rnd < 0.8:
            # 80% 置为 0
            x_masked[idx] = 0
        elif rnd < 0.9:
            # 10% 置为随机值
            x_masked[idx] = torch.randn_like(x_masked[idx], device=device)
        # 其余 10% 保持原样
    return x_masked


def train_masked_node_predictor(
    model: nn.Module,
    data: Data,
    optimizer_class: Type[Optimizer] = torch.optim.Adam,
    optimizer_params: Optional[dict] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: str = "cuda",
    epochs: int = 500,
    warmup_epochs: int = 200,
    initial_lr: float = 1e-4,
    warmup_lr: float = 1e-6,
    smoothing_factor: float = 0.9,
    criterion: Optional[nn.Module] = None,
) -> Tuple[List[float], List[float]]:
    """
    使用 BERT 风格的 Masked Node Prediction 对模型进行训练，并支持自定义优化器、损失函数及学习率调度器。

    参数:
        model (nn.Module): 待训练模型。
        data (Data): torch_geometric.data.Data 对象，必须包含属性 data.x 和 data.edge_index。
        optimizer_class (Type[Optimizer]): 优化器类，默认是 torch.optim.Adam。
        optimizer_params (Optional[dict]): 优化器参数，传递给 optimizer_class 的超参数。
        scheduler (Optional[_LRScheduler]): 学习率调度器；如果为 None，则不使用 scheduler。
        device (str): 设备（默认 "cuda"）。
        epochs (int): 总训练 epoch 数。
        warmup_epochs (int): 热身阶段的 epoch 数，在此阶段采用线性插值调整 lr。
        initial_lr (float): 热身结束时初始学习率。
        warmup_lr (float): 热身开始时的学习率。
        smoothing_factor (float): 指数平滑损失时的平滑因子。
        criterion (Optional[nn.Module]): 损失函数，默认使用 MSELoss。

    返回:
        Tuple[List[float], List[float]]:
            - loss_history: 每个 epoch 的平滑损失列表。
            - lr_history: 每个 epoch 的学习率记录列表。

    异常:
        ValueError: 如果 data 对象不包含必要属性或设备设置有误。
    """
    # 使用默认损失函数
    if criterion is None:
        criterion = nn.MSELoss()

    # 参数检查
    if not hasattr(data, "x"):
        raise ValueError("data 对象必须包含属性 'x'.")
    if not hasattr(data, "edge_index"):
        raise ValueError("data 对象必须包含属性 'edge_index'.")
    if not isinstance(model, nn.Module):
        raise ValueError("model 必须是 torch.nn.Module 的实例.")
    
    # 设置优化器参数
    if optimizer_params is None:
        optimizer_params = {"lr": warmup_lr}

    # 初始化优化器
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    # 转换设备
    device_obj = torch.device(device)
    model.to(device_obj)
    data = data.to(device_obj)

    loss_history: List[float] = []
    lr_history: List[float] = []
    smoothed_loss: Optional[float] = None

    num_nodes = data.x.size(0)

    for epoch in tqdm(range(epochs), desc="Training", leave=True):
        model.train()
        optimizer.zero_grad()

        # 热身阶段的线性学习率调整
        if epoch < warmup_epochs:
            lr = warmup_lr + (initial_lr - warmup_lr) * epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            if scheduler is not None:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        # 生成 mask 张量（保证在同一 device 上）
        mask = torch.rand(num_nodes, device=device_obj) < 0.15
        target = data.x[mask].clone()

        # 对 data.x 进行 masking 操作
        modified_x = apply_mask(data.x, mask, device_obj)
        data.x = modified_x

        # 模型前向传播（假定模型的 forward 接受 data 和 mask）
        try:
            predictions = model(data, mask)
        except Exception as e:
            logging.error("模型前向传播时出错，请检查模型实现和数据格式。")
            raise e

        if predictions.shape != target.shape:
            raise ValueError(
                f"预测结果的 shape {predictions.shape} 与目标 shape {target.shape} 不匹配."
            )

        loss = criterion(predictions, target)

        # 指数平滑损失
        if smoothed_loss is None:
            smoothed_loss = loss.item()
        else:
            smoothed_loss = smoothing_factor * smoothed_loss + (1 - smoothing_factor) * loss.item()

        loss.backward()
        optimizer.step()
        loss_history.append(smoothed_loss)

        if epoch % 50 == 0:
            logging.info(f"Epoch {epoch}/{epochs}, Loss: {smoothed_loss:.4f}, LR: {current_lr:.6f}")

    return loss_history, lr_history