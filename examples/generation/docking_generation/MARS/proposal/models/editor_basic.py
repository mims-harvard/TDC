import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import Set2Set

from MARS.common.nn import GraphEncoder, MLP
from .editor import Editor


class BasicEditor(Editor):
    def __init__(self, config):
        super().__init__(config)
        self.set2set = Set2Set(config["n_node_hidden"], n_iters=6, n_layers=2)
        self.cls_act = MLP(config["n_node_hidden"] * 2, 2)
        self.cls_del = MLP(config["n_node_hidden"] * 2 + config["n_edge_hidden"], 2)
        self.cls_add = MLP(config["n_node_hidden"], 2)
        self.cls_arm = MLP(config["n_node_hidden"], config["vocab_size"])

    def predict_act(self, g, h_node):
        h = self.set2set(g, h_node)
        return self.cls_act(h)

    def predict_del(self, g, h_node):
        with torch.no_grad():
            x_edge = g.edata["e_feat"].to(self.device)

        h_u = h_node[g.edges()[0], :]  # (n_edges, n_node_hidden)
        h_v = h_node[g.edges()[1], :]  # (n_edges, n_node_hidden)
        h_edge = self.edge_mlp(x_edge)
        h_edge = torch.cat(
            [h_u, h_edge, h_v], dim=1
        )  # (n_edges, n_node_hidden*2+n_edge_hidden)
        return self.cls_del(h_edge)

    def predict_add(self, g, h_node):
        return self.cls_add(h_node)

    def predict_arm(self, g, h_node):
        return self.cls_arm(h_node)

    def loss(self, batch, metrics=["loss"]):
        g, edits = batch
        for key in edits.keys():
            edits[key] = edits[key].to(self.device)
        pred_act, pred_del, pred_add, pred_arm = self(g)
        # dist_act = F.softmax(pred_act, dim=1) # (batch_size, 2)
        dist_del = F.softmax(pred_del, dim=1)  # (tot_n_edge, 2)
        dist_add = F.softmax(pred_add, dim=1)  # (tot_n_node, 2)
        dist_arm = F.softmax(pred_arm, dim=1)  # (tot_n_node, vocab_size)
        # loss_act = F.cross_entropy(pred_act, edits['act'])
        # prob_act = dist_act.gather(dim=1,
        #     index=edits['act'].unsqueeze(dim=1)).mean()

        ### targets and masks
        n_node = g.number_of_nodes()
        n_edge = g.number_of_edges()
        off_node, off_edge = 0, 0
        prob_del, prob_add, prob_arm = [], [], []
        with torch.no_grad():
            targ_del = torch.zeros(n_edge).to(self.device).long()
            targ_add = torch.zeros(n_node).to(self.device).long()
            targ_arm = torch.zeros(n_node).to(self.device).long()
            mask_del = torch.zeros(n_edge).to(self.device)
            mask_add = torch.zeros(n_node).to(self.device)
            mask_arm = torch.zeros(n_node).to(self.device)
            for i, g in enumerate(dgl.unbatch(g)):
                n_node = g.number_of_nodes()
                n_edge = g.number_of_edges()
                del_idx = edits["del"][i].item()
                add_idx = edits["add"][i].item()
                arm_idx = edits["arm"][i].item()
                if edits["act"][i].item() == 0:  # del
                    targ_del[off_edge + del_idx] = 1
                    mask_del[off_edge : off_edge + n_edge] = 1.0
                    d_del = dist_del[off_edge : off_edge + n_edge][:, 1]
                    d_del = d_del / (d_del.sum() + 1e-6)
                    prob_del.append(d_del[del_idx].item())
                else:  # add
                    targ_add[off_node + add_idx] = 1
                    targ_arm[off_node + add_idx] = arm_idx
                    mask_add[off_node : off_node + n_node] = 1.0
                    mask_arm[off_node + add_idx] = 1.0
                    d_add = dist_add[off_node : off_node + n_node][:, 1]
                    d_arm = dist_arm[off_node + add_idx]
                    d_add = d_add / (d_add.sum() + 1e-6)
                    d_arm = d_arm / (d_arm.sum() + 1e-6)
                    prob_add.append(d_add[add_idx].item())
                    prob_arm.append(d_arm[arm_idx].item())
                off_node += n_node
                off_edge += n_edge
        prob_del = 1.0 * sum(prob_del) / (len(prob_del) + 1e-6)
        prob_add = 1.0 * sum(prob_add) / (len(prob_add) + 1e-6)
        prob_arm = 1.0 * sum(prob_arm) / (len(prob_arm) + 1e-6)

        ### losses
        loss_del = F.cross_entropy(
            pred_del, targ_del, reduction="none"
        )  # (tot_n_edge,)
        loss_add = F.cross_entropy(
            pred_add, targ_add, reduction="none"
        )  # (tot_n_node,)
        loss_arm = F.cross_entropy(
            pred_arm, targ_arm, reduction="none"
        )  # (tot_n_node,)
        loss_del = (loss_del * mask_del).sum() / (mask_del.sum() + 1e-6)
        loss_add = (loss_add * mask_add).sum() / (mask_add.sum() + 1e-6)
        loss_arm = (loss_arm * mask_arm).sum() / (mask_arm.sum() + 1e-6)
        # loss = loss_act + loss_del + loss_add + loss_arm
        loss = loss_del + loss_add + loss_arm

        local_vars = locals()
        metric_values = [local_vars[metric] for metric in metrics]
        return g.batch_size, metric_values
