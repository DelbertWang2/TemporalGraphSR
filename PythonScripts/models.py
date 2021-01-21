import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class PF_GCN(nn.Module):
    def __init__(self, DAD_matrix):
        super().__init__()
        node_size = DAD_matrix.shape[0]
        self.node_size = node_size
        self.GC1 = GraphConvolution(in_features=64, out_features=128, node_size=node_size, DAD_matrix=DAD_matrix)
        self.GC2 = GraphConvolution(in_features=128, out_features=256, node_size=node_size, DAD_matrix=DAD_matrix)
        self.GC3 = GraphConvolution(in_features=256, out_features=512, node_size=node_size, DAD_matrix=DAD_matrix)
        self.norm4 = nn.BatchNorm1d(33)
        self.GC4 = GraphConvolution(in_features=512, out_features=256, node_size=node_size, DAD_matrix=DAD_matrix)
        self.GC5 = GraphConvolution(in_features=256, out_features=128, node_size=node_size, DAD_matrix=DAD_matrix)
        self.GC6 = GraphConvolution(in_features=128, out_features=64, node_size=node_size, DAD_matrix=DAD_matrix)

    def forward(self, x):
        x = F.relu(self.GC1(x))
        x = F.relu(self.GC2(x))
        x = F.relu(self.GC3(x))
        x = self.norm4(x)
        x = F.relu(self.GC4(x))
        x = F.relu(self.GC5(x))
        x = torch.sigmoid(self.GC6(x))
        return x


class VM_GCN(nn.Module):
    def __init__(self, DAD_matrix):
        super().__init__()
        node_size = DAD_matrix.shape[0]
        self.node_size = node_size
        self.GC1 = GraphConvolution(in_features=64, out_features=128, node_size=node_size, DAD_matrix=DAD_matrix)
        self.GC2 = GraphConvolution(in_features=128, out_features=256, node_size=node_size, DAD_matrix=DAD_matrix)
        self.GC3 = GraphConvolution(in_features=256, out_features=512, node_size=node_size, DAD_matrix=DAD_matrix)
        self.norm4 = nn.BatchNorm1d(33).cuda()
        self.GC4 = GraphConvolution(in_features=512, out_features=256, node_size=node_size, DAD_matrix=DAD_matrix)
        self.GC5 = GraphConvolution(in_features=256, out_features=128, node_size=node_size, DAD_matrix=DAD_matrix)
        self.GC6 = GraphConvolution(in_features=128, out_features=64, node_size=node_size, DAD_matrix=DAD_matrix)

    def forward(self, x):
        x = F.relu(self.GC1(x))
        x = F.relu(self.GC2(x))
        x = F.relu(self.GC3(x))
        x = self.norm4(x)
        x = F.relu(self.GC4(x))
        x = F.relu(self.GC5(x))
        x = torch.sigmoid(self.GC6(x))
        return x


