# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
import os

from .base_group import BenchmarkGroup


class SCDTIGroup(BenchmarkGroup):
    """Create SCDTI Group Class object. This is for single-cell drug-target identification task benchmark.

    Args:
            path (str, optional): the path to store/retrieve the SCDTI group datasets.
    """

    def __init__(self, path="./data", file_format="csv"):
        """Create an SCDTI benchmark group class."""
        # super().__init__(name="SCDTI_Group", path=path)
        self.name = "SCDTI_Group"
        self.path = os.path.join(path, self.name)
        # self.datasets = ["opentargets_dti"]
        self.dataset_names = ["opentargets_dti"]
        self.file_format = file_format
        self.split = None

    def get_train_valid_split(self):
        """parameters included for compatibility. this benchmark has a fixed train/test split."""
        from ..resource.dataloader import DataLoader
        if self.split is None:
            dl = DataLoader(name="opentargets_dti")
            self.split = dl.get_split()
        return self.split["train"], self.split["dev"]

    def get_test(self):
        from ..resource.dataloader import DataLoader
        if self.split is None:
            dl = DataLoader(name="opentargets_dti")
            self.split = dl.get_split()
        return self.split["test"]

    def evaluate(self, y_pred):
        from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
        y_true = self.get_test()["Y"]
        # Calculate metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return [precision, recall, accuracy, f1]

    def evaluate_many(self, preds):
        from numpy import mean, std
        if len(preds) < 5:
            raise Exception(
                "Run your model on at least 5 seeds to compare results and provide your outputs in preds."
            )
        out = dict()
        preds = [self.evaluate(p) for p in preds]
        out["precision"] = (mean([x[0] for x in preds]),
                            std([x[0] for x in preds]))
        out["recall"] = (mean([x[1] for x in preds]), std([x[1] for x in preds
                                                          ]))
        out["accuracy"] = (mean([x[2] for x in preds]),
                           std([x[2] for x in preds]))
        out["f1"] = (mean([x[3] for x in preds]), std([x[3] for x in preds]))
        return out
