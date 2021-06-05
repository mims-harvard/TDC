import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.conv import NNConv

class GraphEncoder(nn.Module):
    def __init__(self, 
            n_atom_feat, n_node_hidden, 
            n_bond_feat, n_edge_hidden, n_layers
        ):
        super().__init__()
        self.embedding = Embedding(n_atom_feat, n_node_hidden, n_bond_feat, n_edge_hidden)
        self.mpnn = MPNN(n_node_hidden, n_edge_hidden, n_layers)

    def forward(self, g, x_node, x_edge):
        '''
        @params:
            g      : batch of dgl.DGLGraph
            x_node : node features, torch.FloatTensor of shape (tot_n_nodes, n_atom_feat)
            x_edge : edge features, torch.FloatTensor of shape (tot_n_edges, n_atom_feat)
        @return:
            h_node : node hidden states
        '''
        h_node, h_edge = self.embedding(g, x_node, x_edge)
        h_node = self.mpnn(g, h_node, h_edge)
        return h_node


class MPNN(nn.Module):
    def __init__(self, n_node_hidden, n_edge_hidden, n_layers):
        super().__init__()
        self.n_layers = n_layers
        edge_network = nn.Sequential(
            nn.Linear(n_edge_hidden, n_edge_hidden), nn.ReLU(),
            nn.Linear(n_edge_hidden, n_node_hidden * n_node_hidden)
        )
        self.conv = NNConv(
            n_node_hidden, n_node_hidden, 
            edge_network, aggregator_type='mean', bias=False)
        self.gru = nn.GRU(n_node_hidden, n_node_hidden)

    def forward(self, g, h_node, h_edge):
        h_gru = h_node.unsqueeze(0)
        for _ in range(self.n_layers):
            m = F.relu(self.conv(g, h_node, h_edge))
            h_node, h_gru = self.gru(m.unsqueeze(0), h_gru)
            h_node = h_node.squeeze(0)
        return h_node


class Embedding(nn.Module):
    def __init__(self, n_atom_feat, n_node_hidden, n_bond_feat, n_edge_hidden):
        super().__init__()
        self.node_emb = nn.Linear(n_atom_feat, n_node_hidden)
        self.edge_emb = nn.Linear(n_bond_feat, n_edge_hidden)

    def forward(self, g, x_node, x_edge):
        h_node = self.node_emb(x_node)
        h_edge = self.edge_emb(x_edge)
        return h_node, h_edge


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=2):
        super().__init__()
        self.out = nn.Linear(in_channels, out_channels)
        self.linears = nn.ModuleList([
            nn.Linear(in_channels, in_channels
            ) for i in range(n_layers)])
        
    def forward(self, x):
        for lin in self.linears:
            x = F.relu(lin(x))
        x = self.out(x)
        return x