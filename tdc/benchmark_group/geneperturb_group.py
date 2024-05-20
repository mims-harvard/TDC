# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
import numpy as np
import os

from .base_group import BenchmarkGroup
from ..dataset_configs.config_map import scperturb_datasets, scperturb_gene_datasets


class GenePerturbGroup(BenchmarkGroup):
    """Create GenePerturbGroup Group Class object. This is for single-cell gene perturbation prediction tasks benchmark.

    Args:
            path (str, optional): the path to store/retrieve the GenePerturb group datasets.
    """

    _GENE_COLS = [
        'UMI_count', 'cancer', 'cell_line', 'disease', 'guide_id', 'ncounts',
        'ngenes', 'nperts', 'organism', 'percent_mito', 'percent_ribo',
        'perturbation', 'perturbation_type', 'tissue_type'
    ]

    def __init__(self, path="./data", file_format="csv"):
        """Create a GenePerturbGroup prediction benchmark group class."""
        self.name = "GenePerturbGroup"
        self.path = os.path.join(path, self.name)
        self.dataset_names = ["scperturb_gene_NormanWeissman2019",
                            "scperturb_gene_ReplogleWeissman2022_rpe1",
                            "scperturb_gene_ReplogleWeissman2022_k562_essential"]
        self.file_format = file_format
        self.split = None

    def get_train_valid_split(self, dataset=None):
        """parameters included for compatibility. this benchmark has a fixed train/test split."""
        from ..multi_pred.perturboutcome import PerturbOutcome
        dataset = dataset or "scperturb_gene_ReplogleWeissman2022_k562_essential"
        assert dataset in self.dataset_names, "{} dataset not in {}".format(
            dataset, self.dataset_names)
        data = PerturbOutcome(dataset)
        self.split = data.get_split()
        
        return self.split[0]["train"], self.split[0]["dev"]

    def get_test(self):
        if self.split is None:
            self.get_train_valid_split()
        return self.split[0]["test"]

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
            r2 = r2_score(df.mean(), y_pred[cell_line].mean())
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
