# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
import os

from .base_group import BenchmarkGroup
from ..resource.pinnacle import PINNACLE


class SCDTIGroup(BenchmarkGroup):
    """Create SCDTI Group Class object. This is for single-cell drug-target identification task benchmark.

    Args:
            path (str, optional): the path to store/retrieve the SCDTI group datasets.
    """

    def __init__(self, path="./data", file_format="csv"):
        """Create an SCDTI benchmark group class."""
        self.name = "SCDTI_Group"
        self.path = os.path.join(path, self.name)
        self.p = PINNACLE()

    def precision_recall_at_k(self, y, preds, k: int = 5):
        """
        Calculate recall@k and precision@k for binary classification.
        """
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, average_precision_score
        assert preds.shape[0] == y.shape[0]
        assert k > 0
        if k > preds.shape[0]:
            return -1, -1, -1, -1

        # Sort the scores and the labels by the scores
        sorted_indices = np.argsort(preds.flatten())[::-1]
        sorted_preds = preds[sorted_indices]
        sorted_y = y[sorted_indices]

        # Get the scores of the k highest predictions
        topk_preds = sorted_preds[:k]
        topk_y = sorted_y[:k]

        # Calculate the recall@k and precision@k
        recall_k = np.sum(topk_y) / np.sum(y)
        precision_k = np.sum(topk_y) / k

        # Calculate the accuracy@k
        accuracy_k = accuracy_score(topk_y, topk_preds > 0.5)

        # Calculate the AP@k
        ap_k = average_precision_score(topk_y, topk_preds)

        return recall_k, precision_k, accuracy_k, ap_k

    def get_train_valid_split(self, seed=1):
        """parameters included for compatibility. this benchmark has a fixed train/test split."""
        train = self.p.get_exp_data(seed=seed, split="train")
        val = self.p.get_exp_data(seed=seed, split="val")
        return {"train": train, "val": val}

    def get_test(self, seed=1):
        return {"test": self.p.get_exp_data(seed=seed, split="test")}

    def evaluate(self, y_pred, k=5, top_k=20, seed=1):
        from numpy import mean
        from sklearn.metrics import roc_auc_score
        y_true = self.get_test(seed=seed)["test"]
        assert "preds" in y_pred.columns, "require 'preds' prediction label in input df"
        assert "cell_type_label" in y_pred.columns, "require cell_type_label in input df"
        assert "disease" in y_pred.columns, "require 'disease' in input df"
        cells = y_true["cell_type_label"].unique()
        diseases = y_true["disease"].unique()
        assert len(cells) == len(
            y_pred["cell_type_label"].unique()
        ), "number of cell types in input df and test df do not match. expected {}".format(
            len(cells))
        assert len(diseases) == len(
            y_pred["disease"].unique()
        ), "number of diseases in input df do not match test df. expected {}".format(
            len(diseases))
        results = {d: [] for d in diseases}
        for disease in diseases:
            for cell in cells:
                preds = y_pred[(y_pred["disease"] == disease) &
                               (y_pred["cell_type_label"] == cell)]
                yt = y_true[(y_true["disease"] == disease) &
                            (y_true["cell_type_label"] == cell)]
                assert len(preds) == len(
                    yt
                ), "mismatch in length of predictions and results for a specific disease {} and cell type {}".format(
                    disease, cell)
                if len(yt) == 0:
                    continue
                auc = roc_auc_score(yt["y"], preds["preds"])
                recall_k, precision_k, accuracy_k, ap_k = self.precision_recall_at_k(
                    yt["y"].values, preds["preds"].values, k=k)
                results[disease].append({
                    "auc": auc,
                    "recall": recall_k,
                    "precision": precision_k,
                    "accuracy": accuracy_k,
                    "ap": ap_k
                })
        # for now, we benchmark with only ap@k with top 20 cells
        for d, scores in results.items():
            assert type(
                scores
            ) == list, "scores should be a list. got {} with value {}".format(
                scores, type(scores))
            assert type(scores[0]
                       ) == dict, "scores should contain dictionary of metrics"
            assert "ap" in scores[0], "scores should include 'ap'"
            topk_cells = [
                x["ap"] for x in sorted(scores, key=lambda s: s["ap"])[-top_k:]
            ]
            results[d] = mean(topk_cells)
        return results

    def evaluate_many(self, preds: list, seed=1):
        from numpy import mean, std
        assert type(
            preds
        ) == list, "expected preds to be a list containing prediction dataframes for multiple seeds"
        if len(preds) < 5:
            raise Exception(
                "Run your model on at least 5 seeds to compare results and provide your outputs in preds."
            )
        evals = [self.evaluate(x, seed=seed) for x in preds]
        diseases = preds[0]["disease"].unique()
        return {
            d: [mean([x[d] for x in evals]),
                std([x[d] for x in evals])] for d in diseases
        }
