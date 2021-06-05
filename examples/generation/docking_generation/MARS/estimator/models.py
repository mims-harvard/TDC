import torch
import torch.nn.functional as F
from torch import nn
from dgl.nn.pytorch.glob import Set2Set

from ..common.nn import GraphEncoder, MLP


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.encoder = GraphEncoder(
            config['n_atom_feat'], config['n_node_hidden'], 
            config['n_bond_feat'], config['n_edge_hidden'], config['n_layers']
        )
        self.set2set = Set2Set(config['n_node_hidden'], n_iters=6, n_layers=2)
        self.classifier = MLP(config['n_node_hidden']*2, 2)

    def forward(self, g):
        with torch.no_grad():
            g = g.to(self.device)
            x_node = g.ndata['n_feat'].to(self.device)
            x_edge = g.edata['e_feat'].to(self.device)

        h = self.encoder(g, x_node, x_edge)
        h = self.set2set(g, h)
        h = self.classifier(h)
        return h
        
    def loss(self, batch, metrics=['loss']):
        '''
        @params:
            batch: batch from the dataset
                g: batched dgl.DGLGraph
                targs: prediction targets
        @returns:
            g.batch_size
            metric_values: cared metric values for
                           training and recording
        '''
        g, targs = batch
        targs = targs.to(self.device)
        logits = self(g) # (batch_size, 2)
        loss = F.cross_entropy(logits, targs)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            true = pred == targs
            acc = true.float().sum() / g.batch_size
            tp = (true * targs).float().sum()
            rec = tp / (targs.long().sum() + 1e-6)
            prec = tp / (pred.long().sum() + 1e-6)
            f1 = 2 * rec * prec / (rec + prec + 1e-6)
            local_vars = locals()
        
        local_vars['loss'] = loss
        metric_values = [local_vars[metric] for metric in metrics]
        return g.batch_size, metric_values
