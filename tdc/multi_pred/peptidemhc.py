# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")
import sys

from ..utils import print_sys
from . import bi_pred_dataset, multi_pred_dataset
from ..metadata import dataset_names


class PeptideMHC(bi_pred_dataset.DataLoader):
    """Data loader class to load datasets in Peptide-MHC Binding Prediction task.
    More info: https://tdcommons.ai/multi_pred_tasks/peptidemhc/

    Task Description: Regression.
                      Given the amino acid sequence of peptide and the pseudo amino acid sequence of MHC,
                      predict the binding affinity.

    Args:
        name (str): the dataset name.
        path (str, optional):
            The path to save the data file, defaults to './data'
        label_name (str, optional):
            For multi-label dataset, specify the label name, defaults to None
        print_stats (bool, optional):
            Whether to print basic statistics of the dataset, defaults to False


    """

    def __init__(self, name, path="./data", label_name=None, print_stats=False):
        """Create Peptide-MHC Prediction dataloader object"""
        super().__init__(
            name,
            path,
            label_name,
            print_stats,
            dataset_names=dataset_names["PeptideMHC"],
        )
        self.entity1_name = "Peptide"
        self.entity2_name = "MHC"
        self.two_types = True

        if print_stats:
            self.print_stats()

        print("Done!", flush=True, file=sys.stderr)
