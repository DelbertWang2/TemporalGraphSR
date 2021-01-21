import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, node_size, DAD_matrix, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.DAD = DAD_matrix
        self.node_size = node_size
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(node_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size = input.shape[0]
        x_list = []
        for ib in range(batch_size):
            x = input[ib]
            x = torch.matmul(self.DAD, x).reshape([self.node_size, -1])
            x = torch.matmul(x, self.weight.t())
            x_list.append(x)
        output = torch.stack(x_list, dim=0) + self.bias
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )