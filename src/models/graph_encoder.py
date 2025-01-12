import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, LayerNorm
from tqdm import tqdm
from torch_geometric.data import Data
import torch.nn.functional as F

class GATv2EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, edge_dim=None, dropout=0.1):
        super(GATv2EncoderLayer, self).__init__()
        self.gatv2 = GATv2Conv(
            in_channels,
            out_channels,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            bias=True,
            residual=True
        )

        # Learnable linear transformation layer
        self.heads_transform = nn.Linear(out_channels * heads, out_channels)

        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

        self.norm1 = LayerNorm(out_channels * heads)  # After GATv2Conv
        self.norm2 = LayerNorm(out_channels)  # After MLP
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, edge_index, edge_attr):
        # GATv2 forward pass
        out = self.gatv2(x, edge_index, edge_attr)
        out = self.norm1(out)  # Layer normalization after GATv2
        out = self.dropout(out)
        out = self.activation(out)

        # Learnable linear transformation
        out = self.heads_transform(out)

        # MLP with residual connection
        residual_mlp = out
        out = self.mlp(out)
        out = out + residual_mlp  # Add residual connection for MLP
        out = self.norm2(out)  # Layer normalization after MLP
        out = self.dropout(out)

        return out


class MaskedNodePredictorWithEncoder(nn.Module):
    def __init__(self, in_features, hidden_channels, edge_dim=None, heads=4, num_encoders=2, dropout=0.1):
        super(MaskedNodePredictorWithEncoder, self).__init__()
        self.encoders = nn.ModuleList()

        # Add encoder layers
        for i in range(num_encoders):
            self.encoders.append(
                GATv2EncoderLayer(
                    in_channels=in_features if i == 0 else hidden_channels,
                    out_channels=hidden_channels,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
            )

        # Prediction MLP
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, in_features),  # Predict original features
        )

    def forward(self, data, mask):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.clone()  # To avoid modifying the original features

        # # Apply masking
        # x[mask] = 0  # Mask features (can use a learnable token here)

        # Pass through encoder layers
        for encoder in self.encoders:
            x = encoder(x, edge_index, edge_attr)

        # Extract embeddings of masked nodes
        masked_embeddings = x[mask]

        # Predict original features
        predictions = self.predictor(masked_embeddings)

        return predictions

