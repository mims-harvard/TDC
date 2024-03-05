# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")
import sys

from ..utils import print_sys
from . import bi_pred_dataset, multi_pred_dataset
from ..metadata import dataset_names


class DrugSyn(multi_pred_dataset.DataLoader):
    """Data loader class to load datasets in Drug Synergy Prediction task.
    More info: https://tdcommons.ai/multi_pred_tasks/drugsyn/

    Task Description: Regression.
                      Given the gene expression of cell lines and two SMILES strings of the drug combos,
                      predict the drug synergy level.

    Args:
        name (str): the dataset name.
        path (str, optional):
            The path to save the data file, defaults to './data'
        print_stats (bool, optional):
            Whether to print basic statistics of the dataset, defaults to False

    """

    def __init__(self, name, path="./data", print_stats=False):
        """Create Drug Synergy Prediction dataloader object"""
        super().__init__(name,
                         path,
                         print_stats,
                         dataset_names=dataset_names["DrugSyn"])

        if print_stats:
            self.print_stats()

        print("Done!", flush=True, file=sys.stderr)
