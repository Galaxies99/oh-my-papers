import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE


class VariationalGCNEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * embedding_dim, cached = True)
        self.conv_mu = GCNConv(2 * embedding_dim, embedding_dim, cached = True)
        self.conv_logstd = GCNConv(2 * embedding_dim, embedding_dim, cached = True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class VariantionalGraphAutoEncoder(VGAE):
    def __init__(self, in_channels, embedding_dim):
        VGAE.__init__(self, VariationalGCNEncoder(in_channels, embedding_dim))

