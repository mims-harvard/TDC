# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted by TDC.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import copy
import numpy as np
from collections import defaultdict

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches, ParamDict

ALGORITHMS = ["ERM", "IRM", "GroupDRO", "CORAL", "MMD", "MTL", "ANDMask"]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.DTI_Encoder()
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )

        self.network = mySequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        from tdc import Evaluator

        self.evaluator = Evaluator(name="PCC")
        self.loss_fct = torch.nn.MSELoss()

    def update(self, minibatches, unlabeled=None):
        all_d = torch.cat([d for d, t, y in minibatches])
        all_t = torch.cat([t for d, t, y in minibatches])
        all_y = torch.cat([y for d, t, y in minibatches])

        y_pred = self.predict(all_d, all_t).reshape(
            -1,
        )
        y_true = all_y.reshape(
            -1,
        )
        loss = self.loss_fct(y_pred, y_true)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        pcc = self.evaluator(
            y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
        )

        return {"loss": loss.item(), "training_pcc": pcc}

    def predict(self, d, t):
        return self.network(d, t)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        loss_fct = torch.nn.MSELoss()
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.0).to(device).requires_grad_()
        loss_1 = loss_fct(
            logits[::2]
            * scale.reshape(
                -1,
            ),
            y[::2].reshape(
                -1,
            ),
        )
        loss_2 = loss_fct(
            logits[1::2]
            * scale.reshape(
                -1,
            ),
            y[1::2].reshape(
                -1,
            ),
        )
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):

        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_d = torch.cat([d for d, t, y in minibatches])
        all_t = torch.cat([t for d, t, y in minibatches])
        all_y = torch.cat([y for d, t, y in minibatches])

        all_logits = self.network(all_d, all_t)
        all_logits_idx = 0

        pred_all = []
        y_all = []

        for i, (d, t, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + d.shape[0]]
            all_logits_idx += d.shape[0]
            nll += self.loss_fct(
                logits.reshape(
                    -1,
                ),
                y.reshape(
                    -1,
                ),
            )
            penalty += self._irm_penalty(logits, y)

            pred_all = (
                pred_all
                + logits.reshape(
                    -1,
                )
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            y_all = (
                y_all
                + y.reshape(
                    -1,
                )
                .cpu()
                .numpy()
                .tolist()
            )

        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        pcc = self.evaluator(y_all, pred_all)

        self.update_count += 1
        return {
            "loss": loss.item(),
            "nll": nll.item(),
            "penalty": penalty.item(),
            "training_pcc": pcc,
        }


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        pred_all = []
        y_all = []

        for m in range(len(minibatches)):
            d, t, y = minibatches[m]
            logits = self.predict(d, t)

            losses[m] = self.loss_fct(
                logits.reshape(
                    -1,
                ),
                y.reshape(
                    -1,
                ),
            )
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

            pred_all = (
                pred_all
                + logits.reshape(
                    -1,
                )
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            y_all = (
                y_all
                + y.reshape(
                    -1,
                )
                .cpu()
                .numpy()
                .tolist()
            )

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        pcc = self.evaluator(y_all, pred_all)

        return {"loss": loss.item(), "training_pcc": pcc}


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        pred_all = []
        y_all = []

        features = [self.featurizer(di, ti) for di, ti, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, _, yi in minibatches]

        for i in range(nmb):
            objective += self.loss_fct(
                classifs[i].reshape(
                    -1,
                ),
                targets[i].reshape(
                    -1,
                ),
            )

            pred_all = (
                pred_all
                + classifs[i]
                .reshape(
                    -1,
                )
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            y_all = (
                y_all
                + targets[i]
                .reshape(
                    -1,
                )
                .cpu()
                .numpy()
                .tolist()
            )

            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        pcc = self.evaluator(y_all, pred_all)

        return {"loss": objective.item(), "penalty": penalty, "training_pcc": pcc}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=True
        )


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=False
        )


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.DTI_Encoder()
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams["nonlinear_classifier"],
        )
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer(
            "embeddings", torch.zeros(num_domains, self.featurizer.n_outputs)
        )

        self.ema = self.hparams["mtl_ema"]
        self.loss_fct = torch.nn.MSELoss()
        from tdc import Evaluator

        self.evaluator = Evaluator(name="PCC")

    def update(self, minibatches, unlabeled=None):
        loss = 0
        pred_all = []
        y_all = []

        for env, (d, t, y) in enumerate(minibatches):
            pred = self.predict(d, t, env)
            loss += self.loss_fct(
                pred.reshape(
                    -1,
                ),
                y.reshape(
                    -1,
                ),
            )

            pred_all = (
                pred_all
                + pred.reshape(
                    -1,
                )
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            y_all = (
                y_all
                + y.reshape(
                    -1,
                )
                .cpu()
                .numpy()
                .tolist()
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        pcc = self.evaluator(y_all, pred_all)

        return {"loss": loss.item(), "training_pcc": pcc}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = (
                self.ema * return_embedding + (1 - self.ema) * self.embeddings[env]
            )

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, d, t, env=None):
        features = self.featurizer(d, t)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None):

        total_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]

        all_d = torch.cat([d for d, t, y in minibatches])
        all_t = torch.cat([t for d, t, y in minibatches])

        all_logits = self.network(all_d, all_t)
        all_logits_idx = 0
        pred_all, y_all = [], []

        for i, (d, t, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + d.shape[0]]
            all_logits_idx += d.shape[0]

            env_loss = self.loss_fct(
                logits.reshape(
                    -1,
                ),
                y.reshape(
                    -1,
                ),
            )

            pred_all = (
                pred_all
                + logits.reshape(
                    -1,
                )
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            y_all = (
                y_all
                + y.reshape(
                    -1,
                )
                .cpu()
                .numpy()
                .tolist()
            )

            total_loss += env_loss

            env_grads = autograd.grad(
                env_loss, self.network.parameters(), retain_graph=True
            )
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        pcc = self.evaluator(y_all, pred_all)

        return {"loss": mean_loss.item(), "training_pcc": pcc}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = mask.sum() / mask.numel()
            param.grad = mask * avg_grad
            param.grad *= 1.0 / (1e-10 + mask_t)

        return 0
