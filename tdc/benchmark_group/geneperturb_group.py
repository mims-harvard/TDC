# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
import numpy as np
import os

from .counterfactual_group import CounterfactualGroup
from ..dataset_configs.config_map import scperturb_datasets, scperturb_gene_datasets


class GenePerturbGroup(CounterfactualGroup):
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
        self.dataset_names = [
            "scperturb_gene_NormanWeissman2019",
            "scperturb_gene_ReplogleWeissman2022_rpe1",
            "scperturb_gene_ReplogleWeissman2022_k562_essential"
        ]
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
