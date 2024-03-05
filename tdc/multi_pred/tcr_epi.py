# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")
import sys

from ..utils import print_sys
from ..utils.load import download_wrapper, pd_load
from . import bi_pred_dataset, multi_pred_dataset
from ..metadata import dataset_names


class TCREpitopeBinding(multi_pred_dataset.DataLoader):
    """Data loader class to load datasets in T cell receptor (TCR) Specificity Prediction Task.
    More info:

    Task Description: Given the TCR and epitope sequence, predict binding probability.

    Args:
        name (str): the dataset name.
        path (str, optional):
            The path to save the data file, defaults to './data'
        print_stats (bool, optional):
            Whether to print basic statistics of the dataset, defaults to False

    """

    def __init__(self, name, path="./data", print_stats=False):
        """Create TCR Specificity Prediction dataloader object"""
        super().__init__(name,
                         path,
                         print_stats,
                         dataset_names=dataset_names["TCREpitopeBinding"])
        self.entity1_name = "TCR"
        self.entity2_name = "Epitope"

        if print_stats:
            self.print_stats()

        print("Done!", flush=True, file=sys.stderr)
