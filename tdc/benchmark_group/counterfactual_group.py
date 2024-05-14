# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
import os

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

    def get_train_valid_split(self, dataset=None, only_seen=False):
        """parameters included for compatibility. this benchmark has a fixed train/test split."""
        from ..multi_pred.perturboutcome import PerturbOutcome
        if only_seen:
            raise ValueError(
                "Counterfactual does not currently support the 'only seen' split"
            )
        dataset = dataset or "scperturb_drug_AissaBenevolenskaya2021"
        assert dataset in self.dataset_names, "{} dataset not in {}".format(
            dataset, self.dataset_names)
        data = PerturbOutcome(dataset)
        self.split = data.get_split()
        return self.split["train"], self.split["dev"]

    def get_test(self):
        if self.split is None:
            self.get_train_valid_split()
        return self.split["test"]

    def evaluate(self, y_pred):
        from sklearn.metrics import r2_score
        y_true = self.get_test()
        cols_to_drop = self._DRUG_COLS if self.is_drug else self._GENE_COLS
        y_true = y_true.drop(cols_to_drop, axis=1)
        y_pred = y_pred.drop(cols_to_drop, axis=1)
        return r2_score(y_true, y_pred)

    def evaluate_dev(
        self, y_pred
    ):  # TODO: under development; benchmark using cell line splits.. will benchmark on random split for now
        from sklearn.metrics import r2_score
        from numpy import average, std
        assert type(
            y_pred
        ) == dict, "evaluate() expects a dictionary with control and perturbation dataframes"
        cols_to_drop = self._DRUG_COLS if self.is_drug else self._GENE_COLS
        y_true = self.get_test()
        # validate input predictions have the same cell lines and perturbations as ground truth
        assert len(y_pred["control"]) == len(
            y_true["control"]
        ), "input pred and ground truth defer in control row ct; {} vs {}".format(
            len(y_pred["control"]), len(y_true["control"]))
        assert y_pred["control"].columns == y_true["control"].columns, \
               "Predictions do not match ground truth columns; lengths are:\n{}\n{}".\
               format(len(y_pred["control"].columns), len(y_true["control"].columns))
        assert len(y_pred["perturbations"]) == len(y_true["perturbations"]), \
            "Perturbation lists do not match length; lengths are:\n{},\n{}".\
            format(len(y_pred["perturbation"]), len(y_true["perturbation"]))
        assert y_pred["perturbations"].columns == y_true["perturbations"].columns, \
            "Perturbation columns do not match; lengths are:\n{},\n{}".\
            format(len(y_pred["perturbations"].columns), len(y_true["perturbations"].columns))
        cell_lines = y_pred["control"]["cell_line"].unique()
        assert set(cell_lines) == set(y_true["control"]["cell_line"].unique()), \
            "Control lines do not match; lengths are:\n{}\n{}".\
                format(len(cell_lines),len(y_true["control"]["cell_line"].unique()))
        cell_lines_perturb = y_pred["perturbations"]["cell_line"].unique()
        assert set(cell_lines_perturb) == set(y_true["perturbations"]["cell_line"].unique()), \
            "Cell lines with perturbations do not match; lengths are:\n{}\n{}".\
                format(len(cell_lines_perturb),len(y_true["perturbations"]["cell_line"].unique()))
        assert set(cell_lines) == set(cell_lines_perturb), \
            "Cell lines do not match; lengths are:\n{}\n{}".\
                format(len(cell_lines),len(cell_lines_perturb))
        r2vec = []
        for line in cell_lines:
            perturbations = y_pred["perturbations"][
                y_pred["perturbations"]["cell_line"] ==
                line]["perturbation"].unique()
            for p in perturbations:
                perturbs_pred = y_pred["perturbations"][
                    y_pred["perturbations"]["cell_line"] == line and
                    y_pred["perturbations"]["perturbation"] == p]
                perturbs_true = y_true["perturbations"][
                    y_true["perturbations"]["cell_line"] == line and
                    y_true["perturbations"]["perturbation"] == p]
                perturbs_pred.drop(cols_to_drop, axis=1, inplace=True)
                perturbs_true.drop(cols_to_drop, axis=1, inplace=True)
                pred_mean = perturbs_pred.mean()
                true_mean = perturbs_true.mean()
                r2vec.append(r2_score(true_mean, pred_mean))
        return {"mean_r2": average(r2vec), "std_r2": std(r2vec)}

    def evaluate_many(self, preds):
        from numpy import mean, std
        if len(preds) < 5:
            raise Exception(
                "Run your model on at least 5 seeds to compare results and provide your outputs in preds."
            )
        out = dict()
        preds = [self.evaluate(p) for p in preds]
        out["mean_R^2"] = mean([x["mean_r2"] for x in preds])
        out["std_R^2"] = mean([x["std_r2"] for x in preds])
        out["seedstd_R^2"] = std([x["mean_r2"] for x in preds])
        return out
