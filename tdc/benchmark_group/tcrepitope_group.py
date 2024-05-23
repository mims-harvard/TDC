# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
import os

from .base_group import BenchmarkGroup


class TCREpitopeGroup(BenchmarkGroup):
    """Create SCDTI Group Class object. This is for single-cell drug-target identification task benchmark.

    Args:
            path (str, optional): the path to store/retrieve the SCDTI group datasets.
    """

    def __init__(self, path="./data", file_format="csv"):
        """Create an SCDTI benchmark group class."""
        # super().__init__(name="SCDTI_Group", path=path)
        self.name = "TCREpitopeGroup"
        self.path = os.path.join(path, self.name)
        # self.datasets = ["opentargets_dti"]
        self.dataset_names = ["tchard"]
        self.file_format = file_format
        self.split = None

    def get_train_valid_split(self):
        """parameters included for compatibility. this benchmark has a fixed train/test split."""
        from ..resource.dataloader import DataLoader
        if self.split is None:
            dl = DataLoader(name="tchard")
            self.split = dl.get_split()
        return self.split["train"], self.split["dev"]

    def get_test(self):
        if self.split is None:
            self.get_train_valid_split()
        return self.split["test"]

    def evaluate(self, y_pred):
        import pandas as pd
        from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
        y_true = self.get_test()
        aucs = []
        # Calculate metrics
        for neg_method, splits in y_true.items():
            for key, df in splits.items():
                assert type(df) == pd.DataFrame, (type(df), df)
                # compute metrics
                true = df["Y"]
                pred = y_pred[neg_method][key]["Y"]
                precision = precision_score(true, pred)
                recall = recall_score(true, pred)
                accuracy = accuracy_score(true, pred)
                f1 = f1_score(true, pred)
                auc = roc_auc_score(true, pred)
                y_pred[neg_method][key]["precision"] = precision
                y_pred[neg_method][key]["recall"] = recall
                y_pred[neg_method][key]["accuracy"] = accuracy
                y_pred[neg_method][key]["f1"] = f1
                y_pred[neg_method][key]["auc"] = auc
                aucs.append((auc, len(df)))
        total_samples = sum(x[1] for x in aucs)
        weighted_sum = sum(a * size for a, size in aucs)
        return weighted_sum / total_samples

    def evaluate_many(self, preds):
        from numpy import mean, std
        if len(preds) < 5:
            raise Exception(
                "Run your model on at least 5 seeds to compare results and provide your outputs in preds."
            )
        weighted_aucs = [self.evaluate(p) for p in preds]
        return {"mean_auc": mean(weighted_aucs), "std_aucs": std(weighted_aucs)}
