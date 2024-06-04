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

    def __init__(self,
                 dataset="scperturb_gene_ReplogleWeissman2022_k562_essential",
                 path="./data",
                 file_format="csv"):
        """Create a GenePerturbGroup prediction benchmark group class."""
        self.name = "GenePerturbGroup"
        self.path = os.path.join(path, self.name)
        self.dataset_names = [
            "scperturb_gene_NormanWeissman2019",
            "scperturb_gene_ReplogleWeissman2022_rpe1",
            "scperturb_gene_ReplogleWeissman2022_k562_essential"
        ]
        self.file_format = file_format

        from ..multi_pred.perturboutcome import PerturbOutcome
        dataset = dataset
        assert dataset in self.dataset_names, "{} dataset not in {}".format(
            dataset, self.dataset_names)
        data = PerturbOutcome(dataset)
        self.split = data.get_split()
        self.test = self.get_test()

    def get_train_valid_split(self):
        """parameters included for compatibility. this benchmark has a fixed train/test split."""
        train = self.split[0]["train"]
        val = self.split[0]["dev"]
        self.ctrl = np.array(
            np.mean(train[train.obs.condition == 'ctrl'].X,
                    axis=0)).reshape(-1,)
        return train, val

    def get_test(self):
        return self.split[0]["test"]

    def evaluate(self, pred):
        # pred is a dictionary with perturbation name in the format of X+X, where single gene is X+ctrl
        from scipy.stats import pearsonr
        test = self.test
        ctrl = self.ctrl
        mean_truth = {}
        for condition in test.obs.condition.unique():
            out = test[test.obs.condition == condition].X
            mean_truth[condition] = np.array(np.mean(out, axis=0)).reshape(-1,)

        pearson_delta = []
        for i, j in mean_truth.items():
            pearson_delta.append(pearsonr(pred[i] - ctrl, j - ctrl)[0])

        pearson_delta = np.mean(pearson_delta)

        A = test.var.gene_name.values
        de_20 = {
            i.split('_')[1]:
                np.array(
                    [np.where(A == b)[0][0] if b in A else -1 for b in j[:20]])
            for i, j in test.uns['rank_genes_groups_cov_all'].items()
        }
        mse_20_de = []
        for i, j in mean_truth.items():
            mse_20_de.append(np.mean((pred[i][de_20[i]] - j[de_20[i]])**2))
        mse_20_de = np.mean(mse_20_de)

        direction = []
        for i, j in mean_truth.items():
            direction.append(
                sum(
                    np.sign(pred[i][de_20[i]] -
                            ctrl[de_20[i]]) == np.sign(j[de_20[i]] -
                                                       ctrl[de_20[i]])) / 20)
        direction = np.mean(direction)

        return {
            'pearson_delta_all_genes': pearson_delta,
            'mse_20_de_genes': mse_20_de,
            'ratio_direction_correct_20_de_genes': direction
        }
