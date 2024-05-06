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
        # self.datasets = ["pinnacle_dti"]
        self.dataset_names = ["pinnacle_dti"]
        self.file_format = file_format
        self.split = None

    def get_train_valid_split(self):
        """parameters included for compatibility. this benchmark has a fixed train/test split."""
        from ..resource.dataloader import DataLoader
        if self.split is None:
            dl = DataLoader(name="pinnacle_dti")
            self.split = dl.get_split()
        return self.split["train"], self.split["dev"]

    def get_test(self):
        from ..resource.dataloader import DataLoader
        if self.split is None:
            dl = DataLoader(name="pinnacle_dti")
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
