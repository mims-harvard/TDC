# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import sys

from ..utils import print_sys
from .single_cell import CellXGeneTemplate


class PerturbOutcome(CellXGeneTemplate):

    def __init__(self, name, path="./data", print_stats=False):
        super().__init__(name, path, print_stats)

    def get_mean_expression(self):
        raise ValueError("TODO")

    def get_DE_genes(self):
        raise ValueError("TODO")

    def get_dropout_genes(self):
        raise ValueError("TODO")

    def get_split(self,
                  ratios=[0.8, 0.1, 0.1],
                  unseen=False,
                  use_random=True,
                  random_state=42):
        """obtain train/dev/test splits for each cell_line
        counterfactual prediction model is trained on a single cell line and then evaluated on same cell line
        and against new cell lines
        TODO: also allow for splitting by unseen perturbations
        TODO: allow for evaluating within the same cell line"""
        # For now, we will ensure there are no unseen perturbations
        if unseen:
            raise ValueError(
                "Unseen perturbation splits are not yet implemented!")
        df = self.get_data()
        if use_random:
            # just do a random split, otherwise you'll split by cell line...
            from sklearn.model_selection import train_test_split
            control = df[df["perturbation"] == "control"]
            perturbs = df[df["perturbation"] != "control"]
            train, tmp = train_test_split(perturbs,
                                          test_size=ratios[1] + ratios[2],
                                          random_state=random_state)
            test, dev = train_test_split(tmp,
                                         test_size=ratios[2] /
                                         (ratios[1] + ratios[2]),
                                         random_state=random_state)
            return {
                "control": control,
                "train": train,
                "dev": dev,
                "test": test
            }
        cell_lines = df["cell_line"].unique()
        perturbations = df["perturbation"].unique()
        shuffled_cell_line_idx = np.random.permutation(len(cell_lines))
        assert len(shuffled_cell_line_idx) == len(cell_lines)
        assert len(shuffled_cell_line_idx) > 3

        # Split indices into three parts
        train_end = int(ratios[0] * len(cell_lines))  # 60% for training
        dev_end = train_end + int(
            ratios[1] * len(cell_lines))  # 20% for development

        train_cell_line = shuffled_cell_line_idx[:train_end]
        dev_cell_line = shuffled_cell_line_idx[train_end:dev_end]
        test_cell_line = shuffled_cell_line_idx[dev_end:]

        assert len(train_cell_line) > 0
        assert len(dev_cell_line) > 0
        assert len(test_cell_line) > 0
        assert len(test_cell_line) > len(dev_cell_line)

        train_control = df[(df["cell_line"].isin(train_cell_line)) &
                           (df["perturbation"] == "control")]
        train_perturbations = df[(df["cell_line"].isin(train_cell_line)) &
                                 (df["perturbation"] != "control")]

        assert len(train_control) > 0
        assert len(train_perturbations) > 0
        assert len(train_control) <= len(train_perturbations)

        dev_control = df[(df["cell_line"].isin(dev_cell_line)) &
                         (df["perturbation"] == "control")]
        dev_perturbations = df[(df["cell_line"].isin(dev_cell_line)) &
                               (df["perturbation"] != "control")]

        test_control = df[(df["cell_line"].isin(test_cell_line)) &
                          (df["perturbation"] == "control")]
        test_perturbations = df[(df["cell_line"].isin(test_cell_line)) &
                                (df["perturbation"] != "control")]

        out = {}
        out["train"] = {
            "control": train_control,
            "perturbations": train_perturbations
        }
        out["dev"] = {
            "control": dev_control,
            "perturbations": dev_perturbations
        }
        out["test"] = {
            "control": test_control,
            "perturbations": test_perturbations
        }
        # TODO: currently, there will be no inter-cell-line evaluation
        return out
