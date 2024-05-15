# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
import os

from .base_group import BenchmarkGroup


class ProteinPeptideGroup(BenchmarkGroup):
    """Create Protein-Peptide Group Class object. This is for benchmarking models predicting protein-peptide interactions.

    Args:
            path (str, optional): the path to store/retrieve the Protein-Peptide group datasets.
    """

    def __init__(self, path="./data", file_format="csv"):
        """Create an SCDTI benchmark group class."""
        # super().__init__(name="SCDTI_Group", path=path)
        self.name = "ProteinPeptide_Group"
        self.path = os.path.join(path, self.name)
        # self.datasets = ["opentargets_dti"]
        self.dataset_names = ["brown_mdm2_ace2_12ca5"]
        self.file_format = file_format
        self.split = None

    def get_train_valid_split(self):
        """parameters included for compatibility. this benchmark has a fixed train/test split."""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from ..multi_pred.proteinpeptide import ProteinPeptide as DataLoader
        if self.split is None:
            dl = DataLoader(name="brown_mdm2_ace2_12ca5")
            df = dl.get_data()
            for idx, e in enumerate(df["Y"]):
                if e != "Putative binder":
                    df["Y"][idx] = "1"
                else:
                    df["Y"][idx] = "0"
            # raise Exception("unique", )
            # Split the data while stratifying
            X_train, X_test, y_train, y_test = train_test_split(
                df.drop('Y', axis=1),  # features
                df['Y'],  # labels
                test_size=0.9,  # 90% of the data goes to the test set
                random_state=42,  # for reproducibility
                stratify=df[
                    'Y']  # stratify by the label column to ensure even distribution
            )
            self.split = {}
            self.split["train"] = (X_train, y_train)
            self.split["test"] = (X_test, y_test)
            self.split["dev"] = []

        return self.split["train"], self.split["dev"]

    def get_test(self):
        from ..multi_pred.proteinpeptide import ProteinPeptide as DataLoader
        if self.split is None:
            self.get_train_valid_split()
        return self.split["test"]

    def evaluate(self, y_pred):
        from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
        y_true = self.get_test()[1]
        # Calculate metrics
        precision = precision_score(y_true, y_pred, pos_label="1")
        recall = recall_score(y_true, y_pred, pos_label="1")
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, pos_label="1")
        auc = roc_auc_score(y_true, y_pred)
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
        # out["auc"] = (mean([x[4] for x in preds]), std([x[4] for x in preds]))
        return out
