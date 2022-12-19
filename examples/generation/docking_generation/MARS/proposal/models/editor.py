import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from MARS.common.nn import GraphEncoder, MLP


class Editor(ABC, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.encoders = nn.ModuleDict()
        for name in ["act", "del", "add", "arm"]:
            encoder = GraphEncoder(
                config["n_atom_feat"],
                config["n_node_hidden"],
                config["n_bond_feat"],
                config["n_edge_hidden"],
                config["n_layers"],
            )
            self.encoders[name] = encoder
        self.edge_mlp = MLP(config["n_bond_feat"], config["n_edge_hidden"])

    @abstractmethod
    def predict_act(self, g, h_node):
        raise NotImplementedError

    @abstractmethod
    def predict_del(self, g, h_node):
        raise NotImplementedError

    @abstractmethod
    def predict_add(self, g, h_node):
        raise NotImplementedError

    @abstractmethod
    def predict_arm(self, g, h_node):
        raise NotImplementedError

    def forward(self, g):
        """
        @params:
            g: dgl.batch of molecular skeleton graphs
        @return:
            pred_act: action (del or add) prediction, torch.FloatTensor of shape (batch_size, 2)
            pred_del: bond breaking prediction,       torch.FloatTensor of shape (tot_n_edge, 2)
            pred_add: place to add arm,               torch.FloatTensor of shape (tot_n_node, 2)
            pred_arm: arm from vocab to add,          torch.FloatTensor of shape (tot_n_node, vocab_size)
        """
        with torch.no_grad():
            g = g.to(self.device)
            x_node = g.ndata["n_feat"].to(self.device)
            x_edge = g.edata["e_feat"].to(self.device)

        ### encode graph nodes
        encoded = {}
        for name, encoder in self.encoders.items():
            # h_node, (tot_n_node, n_node_hidden)
            encoded[name] = encoder(g, x_node, x_edge)

        # pred_act = self.predict_act(g, encoded['act'])
        pred_del = self.predict_del(g, encoded["del"])
        pred_add = self.predict_add(g, encoded["add"])
        pred_arm = self.predict_arm(g, encoded["arm"])
        # return pred_act, pred_del, pred_add, pred_arm
        return None, pred_del, pred_add, pred_arm
