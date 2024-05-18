# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import train_test_split
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

    def get_cellline_split(self,
                           ratios=[0.8, 0.1, 0.1],
                           random_state=42,
                           split_to_unseen=False):
        df = self.get_data()
        print("got data grouping by cell line")
        cell_line_groups = df.groupby("cell_line")
        print("groupby completed")
        cell_line_splits = {}
        for cell_line, cell_line_group in cell_line_groups:
            print("processing cell line", cell_line)
            control = cell_line_group[cell_line_group["perturbation"] ==
                                      "control"]
            cell_line_group = cell_line_group[cell_line_group["perturbation"] !=
                                              "control"]
            if not split_to_unseen:
                train, tmp = train_test_split(cell_line_group,
                                              test_size=ratios[1] + ratios[2],
                                              random_state=random_state)
                test, dev = train_test_split(tmp,
                                             test_size=ratios[2] /
                                             (ratios[1] + ratios[2]),
                                             random_state=random_state)
                cell_line_splits[cell_line] = {
                    "control": control,
                    "train": train,
                    "test": test,
                    "dev": dev
                }
            else:
                perturbs = cell_line_group["perturbation"].unique()
                perturbs_train, tmp = train_test_split(
                    perturbs,
                    test_size=ratios[1] + ratios[2],
                    random_state=random_state)
                perturbs_test, perturbs_dev = train_test_split(
                    tmp,
                    test_size=ratios[2] / (ratios[1] + ratios[2]),
                    random_state=random_state)
                cell_line_splits[cell_line] = {
                    "control":
                        control,
                    "train":
                        cell_line_group[
                            cell_line_group["perturbation"].isin(perturbs_train)
                        ],
                    "test":
                        cell_line_group[
                            cell_line_group["perturbation"].isin(perturbs_test)
                        ],
                    "dev":
                        cell_line_group[
                            cell_line_group["perturbation"].isin(perturbs_dev)]
                }
            print("done with cell line", cell_line)

        return cell_line_splits

    def get_split(self,
                  ratios=[0.8, 0.1, 0.1],
                  unseen=False,
                  use_random=False,
                  random_state=42):
        """obtain train/dev/test splits for each cell_line
        counterfactual prediction model is trained on a single cell line and then evaluated on same cell line
        and against new cell lines
        TODO: also allow for splitting by unseen perturbations
        TODO: allow for evaluating within the same cell line"""
        if not use_random:
            return self.get_cellline_split(split_to_unseen=unseen,
                                           ratios=ratios,
                                           random_state=random_state)
        df = self.get_data()
        # just do a random split, otherwise you'll split by cell line...
        control = df[df["perturbation"] == "control"]
        perturbs = df[df["perturbation"] != "control"]
        train, tmp = train_test_split(perturbs,
                                      test_size=ratios[1] + ratios[2],
                                      random_state=random_state)
        test, dev = train_test_split(tmp,
                                     test_size=ratios[2] /
                                     (ratios[1] + ratios[2]),
                                     random_state=random_state)
        return {"control": control, "train": train, "dev": dev, "test": test}
