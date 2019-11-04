import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder with ReLU activation, independent of graph structure.
    """
    def __init__(
            self,
            node_dim=2,    
            embed_dim=256,
            n_layers=3,
            **kwargs
    ):
        super(MLPEncoder, self).__init__()
        # Define net parameters
        self.node_dim = node_dim
        self.hidden_dim = embed_dim
        self.num_layers = n_layers
        # Define GCN Layers
        self.mlp_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, True) for _ in range(self.num_layers)])

    def forward(self, x):
        """
        Args:
            input: Input nodes (batch_size, num_nodes, hidden_dim)
        """
        for layer in range(self.num_layers):
            x = self.mlp_layers[layer](x)  # B x V x H
            x = F.relu(x)
        
        return {
            "node_embs": x, # (batch_size, num_nodes, hidden_dim)
            "graph_embs": x.mean(dim=1),  # (batch_size, hidden_dim)
        }
