import copy
from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.data import Data


class FineTuningModelBertStyle(nn.Module):
    """
    该模块用于复制预训练的 MaskedNodePredictorWithEncoder 模型中的 encoder 部分，
    并在 encoder 后接上自定义的输出层，用于针对其他任务的微调。

    参数:
        pretrained_model (nn.Module): 预训练的 MaskedNodePredictorWithEncoder 模型。
        output_head (nn.Module): 自定义的输出层模块，可以为 MLP、GNN 等。
        freeze_encoder (bool): 是否冻结 encoder 部分参数，默认 False，设置为 True 则仅训练输出层。
    """
    def __init__(self, pretrained_model: nn.Module, output_head: nn.Module, freeze_encoder: bool = False):
        super(FineTuningModelBertStyle, self).__init__()

        # 检查 pretrained_model 是否具有 encoder 属性
        if not hasattr(pretrained_model, "encoders"):
            raise ValueError("预训练模型必须包含属性 'encoders'.")

        # 复制 encoder 部分（深拷贝保证与预训练模型参数独立）
        self.encoders = copy.deepcopy(pretrained_model.encoders)
        
        # 保存自定义输出层
        self.output_head = output_head

        # 根据需求设置是否冻结 encoder 参数
        if freeze_encoder:
            for param in self.encoders.parameters():
                param.requires_grad = False

    def forward(self, data: Data, mask: Optional[torch.Tensor] = None):
        """
        前向传播：
            - data: 输入的 torch_geometric.data.Data 对象（必须包含 x、edge_index、edge_attr 等）。
            - mask: 可选的 Boolean mask，用于从 encoder 输出中选取特定节点的表示；如果为 None，则对所有节点输出进行后续处理。

        返回:
            输出层的计算结果，可根据自定义输出层的定义而定。
        """
        # 提取输入特征及图数据
        if not (hasattr(data, "x") and hasattr(data, "edge_index")):
            raise ValueError("输入数据必须包含 'x' 和 'edge_index' 属性。")

        x = data.x.clone()
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        # 将特征通过复制的 encoder 层
        for encoder in self.encoders:
            # 如果 encoder 需要 edge_attr，可直接传入；否则也能正常运行
            x = encoder(x, edge_index, edge_attr)
        
        # 如果提供了 mask，则仅选择对应节点的输出
        if mask is not None:
            x = x[mask]
        
        # 最后传入自定义输出层获得最终结果
        out = self.output_head(x)
        return out