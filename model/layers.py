import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn


class GraphConvolution(Module):
    """
    LGG-specific GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class GCN(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):
        graph, adj = data
        adj = self.norm_adj(adj)
        support = torch.matmul(graph, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = (F.relu(output + self.bias), adj)
        else:
            output = (F.relu(output), adj)
        return output

    def norm_adj(self, adj):
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj


class ChebyNet(Module):
    def __init__(self, K, in_feature, out_feature):
        super(ChebyNet, self).__init__()
        self.K = K
        self.filter_weight, self.filter_bias = self.init_fliter(K, in_feature, out_feature)

    def init_fliter(self, K, feature, out, bias=True):
        weight = nn.Parameter(torch.FloatTensor(K, 1, feature, out), requires_grad=True)
        nn.init.normal_(weight, 0, 0.1)
        bias_ = None
        if bias == True:
            bias_ = nn.Parameter(torch.zeros((1, 1, out), dtype=torch.float32), requires_grad=True)
            nn.init.normal_(bias_, 0, 0.1)
        return weight, bias_

    def get_L(self, adj):
        #device = adj.device
        degree = torch.sum(adj, dim=1)
        degree_norm = torch.div(1.0, torch.sqrt(degree) + 1.0e-5)
        degree_matrix = torch.diag(degree_norm)
        #I = torch.eye(degree_matrix.size(0), degree_matrix.size(1)).to(device)
        L = - torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
        return L

    # def rescale_L(self, L):
    #     largest_eigval, _ = torch.symeig(L, eigenvectors=True)
    #     largest_eigval = torch.max(largest_eigval)
    #     L = L - torch.eye(L.size(0), device=L.device, dtype=torch.float)
    #     return L

    def chebyshev(self, x, L):
        # to do graph convolution here] X_0 = X, X_1 = L.X, X_k = 2.L.X_(k-1) - X_(k-2)
        x1 = torch.matmul(L, x)
        x_ = torch.stack((x, x1), dim=1)  # (b, 2, chan, fin)
        if self.K > 1:
            for k in range(2, self.K):
                x_current = 2 * torch.matmul(L, x_[:, -1]) - x_[:, -2]  # X_k = 2.L.X_(k-1) - X_(k-2)
                x_current = x_current.unsqueeze(dim=1)
                x_ = torch.cat((x_, x_current), dim=1)

        x_ = x_.permute(1, 0, 2, 3)  # (k, b, chan, fin)   w: (k, 1, fin, fout) f:
        out = torch.matmul(x_, self.filter_weight)  # (k, b, chan, fout)
        out = torch.sum(out, dim=0)  # (b, chan, fout)
        out = F.relu(out + self.filter_bias)
        return out

    def forward(self, data):
        # x: (b, chan, f) adj
        x, adj = data
        L = self.get_L(adj)
        out = self.chebyshev(x, L)
        out = (out, adj)
        return out

