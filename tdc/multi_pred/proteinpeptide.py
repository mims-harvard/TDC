# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")
import sys

from ..utils import print_sys
from . import bi_pred_dataset
from ..metadata import dataset_names
from ..dataset_configs.config_map import ConfigMap


class ProteinPeptide(bi_pred_dataset.DataLoader):
    """Data loader class to load datasets in Protein-Peptide Binding Prediction task.
    More info: TODO 

    Task Description: Regression.
                      Given the amino acid sequence of peptide and (TODO: complete),
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
        """Create Protein-Peptide Prediction dataloader object"""
        label_name = label_name if label_name is not None else "KD (nm)"  # TODO: this column should be parsed into float and upper/lower
        cfm = ConfigMap()
        config = cfm.get(name)
        super().__init__(
            name,
            path,
            label_name,
            print_stats,
            dataset_names=dataset_names["ProteinPeptide"],
            data_config=config(),
        )
        self.entity1_name = "Sequence"  # peptide sequence
        self.entity2_name = "Protein Target"  # protein target label
        self.two_types = True

        if print_stats:
            self.print_stats()

        print("Done!", flush=True, file=sys.stderr)
