#This is the script of Graph in Graph (GiG) for continous emotion recognition
from model.layers import ChebyNet, GCN
import torch
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn


class GraphEncoder(nn.Module):
    def __init__(self, num_layers, num_node, in_features, out_features, K,
                 encoder_type='GCN'):
        super(GraphEncoder, self).__init__()
        self.K = K  # useful for ChebyNet
        self.tokenizer = nn.Linear(num_node*out_features, out_features)
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = self.get_layer(encoder_type, in_features, out_features)
            else:
                layer = self.get_layer(encoder_type, out_features, out_features)
            layers.append(layer)
        self.encoder = nn.Sequential(*layers)

    def get_layer(self, encoder_type, in_features, out_features):
        assert encoder_type in ['Cheby', 'GCN'], "encoder type is not supported!"
        if encoder_type == 'GCN':
            GNN = GCN(in_features, out_features)
        if encoder_type == 'Cheby':
            GNN = ChebyNet(self.K, in_features, out_features)
        return GNN

    def forward(self, x, adj, flatten=True):
        # x: b, channel, feature
        # adj: m, n, n
        output = self.encoder((x, adj))
        x, _ = output
        if flatten:
            x = x.view(x.size(0), -1)
            output = self.tokenizer(x)
        else:
            output = x
        return output


class GiG(nn.Module):
    def __init__(self, layers_graph=[1, 2], num_chan=62, num_seq=96,
                 num_feature=5, hidden_graph=64, K=[2, 4],
                 dropout=0.25, num_class=1, encoder_type='GCN'):
        super(GiG, self).__init__()
        self.graph_encoder_type = encoder_type
        self.SGE = GraphEncoder(
            num_layers=layers_graph[0], num_node=num_chan, in_features=num_feature,
            out_features=hidden_graph, K=K[0], encoder_type=encoder_type
        )

        self.s_adj = nn.Parameter(torch.FloatTensor(num_chan, num_chan), requires_grad=True)
        nn.init.xavier_uniform_(self.s_adj)

        self.t_adj = nn.Parameter(torch.FloatTensor(num_seq, num_seq), requires_grad=True)
        nn.init.xavier_uniform_(self.t_adj)

        self.TGE = GraphEncoder(
            num_layers=layers_graph[1], num_node=num_seq, in_features=hidden_graph,
            out_features=hidden_graph, K=K[1], encoder_type=encoder_type
        )

        self.MLP = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_graph, num_class)
        )

    def forward(self, x):
        # x: batch, sequence, chan, feature
        x = x[list(x.keys())[0]]
        b, s, chan, f = x.size()
        x = rearrange(x, 'b s c f  -> (b s) c f')
        s_adj = self.get_adj(self.s_adj)
        t_adj = self.get_adj(self.t_adj)

        # spatial GCN first
        x = self.SGE(x, s_adj)
        # temporal contextual transformer
        x = rearrange(x, '(b s) h -> b s h', b=b, s=s)

        # temporal GCN here
        x = self.TGE(x, t_adj, False)
        x = self.MLP(x)
        return x

    def get_adj(self, adj):
        # adj : node, node
        num_nodes = adj.shape[-1]
        adj = F.relu(adj + adj.transpose(1, 0))
        if self.graph_encoder_type == 'GCN':
            adj = adj + torch.eye(num_nodes).to(adj.device)
        return adj


class iG(nn.Module):
    def __init__(self, layers_graph=[1, 2], num_chan=62, num_seq=96,
                 num_feature=5, hidden_graph=64, K=[2, 4],
                 dropout=0.25, num_class=1, encoder_type='GCN'):
        super(iG, self).__init__()
        self.graph_encoder_type = encoder_type
        self.to_node = nn.Linear(num_chan*num_feature, hidden_graph, bias=False)

        self.t_adj = nn.Parameter(torch.FloatTensor(num_seq, num_seq), requires_grad=True)
        nn.init.xavier_uniform_(self.t_adj)

        self.TGE = GraphEncoder(
            num_layers=layers_graph[1], num_node=num_seq, in_features=hidden_graph,
            out_features=hidden_graph, K=K[1], encoder_type=encoder_type
        )

        self.MLP = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_graph, num_class)
        )

    def forward(self, x):
        # x: batch, sequence, chan, feature
        x = x[list(x.keys())[0]]
        b, s, chan, f = x.size()
        x = rearrange(x, 'b s c f  -> b s (c f)')
        x = self.to_node(x)   # (b, s, h)

        t_adj = self.get_adj(self.t_adj)

        # temporal GCN here
        x = self.TGE(x, t_adj, False)
        x = self.MLP(x)
        return x

    def get_adj(self, adj):
        # adj : node, node
        num_nodes = adj.shape[-1]
        adj = F.relu(adj + adj.transpose(1, 0))
        if self.graph_encoder_type == 'GCN':
            adj = adj + torch.eye(num_nodes).to(adj.device)
        return adj

class Gi(nn.Module):
    def __init__(self, layers_graph=[1, 2], num_chan=62, num_seq=96,
                 num_feature=5, hidden_graph=64, K=[2, 4],
                 dropout=0.25, num_class=1, encoder_type='GCN'):
        super(Gi, self).__init__()
        self.graph_encoder_type = encoder_type
        self.SGE = GraphEncoder(
            num_layers=layers_graph[0], num_node=num_chan, in_features=num_feature,
            out_features=hidden_graph, K=K[0], encoder_type=encoder_type
        )

        self.s_adj = nn.Parameter(torch.FloatTensor(num_chan, num_chan), requires_grad=True)
        nn.init.xavier_uniform_(self.s_adj)

        self.MLP = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_graph, num_class)
        )

    def forward(self, x):
        # x: batch, sequence, chan, feature
        x = x[list(x.keys())[0]]
        b, s, chan, f = x.size()
        x = rearrange(x, 'b s c f  -> (b s) c f')
        s_adj = self.get_adj(self.s_adj)

        # spatial GCN first
        x = self.SGE(x, s_adj)
        # temporal contextual transformer
        x = rearrange(x, '(b s) h -> b s h', b=b, s=s)
        x = self.MLP(x)
        return x

    def get_adj(self, adj):
        # adj : node, node
        num_nodes = adj.shape[-1]
        adj = F.relu(adj + adj.transpose(1, 0))
        if self.graph_encoder_type == 'GCN':
            adj = adj + torch.eye(num_nodes).to(adj.device)
        return adj
