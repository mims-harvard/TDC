# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
import numpy as np
import os
import pandas as pd

from .base_group import BenchmarkGroup
from ..dataset_configs.config_map import scperturb_datasets, scperturb_gene_datasets


class CounterfactualGroup(BenchmarkGroup):
    """Create Counterfactual Group Class object. This is for single-cell counterfactual prediction tasks (drug, gene) benchmark.

    Args:
            path (str, optional): the path to store/retrieve the Counterfactual group datasets.
    """
    _DRUG_COLS = [
        "ncounts", 'celltype', 'cell_line', 'cancer', 'disease', 'tissue_type',
        'perturbation', 'perturbation_type', 'ngenes'
    ]

    _GENE_COLS = [
        'UMI_count', 'cancer', 'cell_line', 'disease', 'guide_id', 'ncounts',
        'ngenes', 'nperts', 'organism', 'percent_mito', 'percent_ribo',
        'perturbation', 'perturbation_type', 'tissue_type'
    ]

    def __init__(self, path="./data", file_format="csv", is_drug=True):
        """Create a Counterfactual prediction benchmark group class."""
        self.name = "Coutnerfactual_Group"
        self.path = os.path.join(path, self.name)
        self.is_drug = is_drug
        self.dataset_names = scperturb_gene_datasets if not self.is_drug else scperturb_datasets
        self.file_format = file_format
        self.split = None

    def get_train_valid_split(self,
                              dataset=None,
                              split_to_unseen=False,
                              remove_unseen=True):
        """parameters included for compatibility. this benchmark has a fixed train/test split."""
        from ..multi_pred.perturboutcome import PerturbOutcome
        dataset = dataset or "scperturb_drug_AissaBenevolenskaya2021"
        assert dataset in self.dataset_names, "{} dataset not in {}".format(
            dataset, self.dataset_names)
        data = PerturbOutcome(dataset)
        self.split = data.get_split(unseen=split_to_unseen,
                                    remove_unseen=remove_unseen)
        cell_lines = list(self.split.keys())
        self.split["adj"] = 0
        for line in cell_lines:
            print("processing benchmark line", line)
            for split, df in self.split[line].items():
                if split not in self.split:
                    self.split[split] = {}
                elif split == "adj":
                    self.split["adj"] += df
                    continue
                self.split[split][line] = df
            print("done with line", line)
        return self.split["train"], self.split["dev"]

    def get_test(self):
        if self.split is None:
            self.get_train_valid_split()
        return self.split["test"]

    def evaluate(self, y_pred):
        from sklearn.metrics import r2_score
        y_true = self.get_test()
        r2vec = []
        for cell_line, df in y_true.items():
            check = self._DRUG_COLS[0] if self.is_drug else self._GENE_COLS[0]
            cols = self._DRUG_COLS if self.is_drug else self._GENE_COLS
            if check in df.columns:
                df.drop(cols, axis=1)
            if check in y_pred[cell_line].columns:
                y_pred[cell_line].drop(cols, axis=1)
            categorical_cols = df.select_dtypes(
                include=['object', 'category']).columns
            df = df.drop(columns=categorical_cols)
            categorical_cols = y_pred[cell_line].select_dtypes(
                include=['object', 'category']).columns
            y_pred[cell_line] = y_pred[cell_line].drop(columns=categorical_cols)
            mdf = df.mean()
            mpred = y_pred[cell_line].mean()
            if len(mdf) != len(mpred):
                raise Exception(
                    "lengths between true and test mean vectors defers in cell line {} with {} vs {}"
                    .format(cell_line, len(mdf), len(mpred)))
            elif pd.isna(mdf.values).any():
                raise Exception(
                    "ground truth mean contains {} nan values".format(
                        mdf.isna().sum()))
            elif pd.isna(mpred.values).any():
                raise Exception("prediction mean contains {} nan values".format(
                    mpred.isna().sum()))
            r2 = r2_score(mdf, mpred)
            r2vec.append(r2)
        return np.mean(r2vec)

    def evaluate_many(self, preds):
        from numpy import mean, std
        if len(preds) < 5:
            raise Exception(
                "Run your model on at least 5 seeds to compare results and provide your outputs in preds."
            )
        out = dict()
        preds = [self.evaluate(p) for p in preds]
        out["mean_R^2"] = mean(preds)
        out["std_R^2"] = std(preds)
        return out
