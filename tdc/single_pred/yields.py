# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import sys
import warnings

warnings.filterwarnings("ignore")

from . import single_pred_dataset
from ..utils import print_sys
from ..metadata import dataset_names


class Yields(single_pred_dataset.DataLoader):
    """Data loader class to load datasets in Yields (Reaction Yields Prediction) task. More info: https://tdcommons.ai/single_pred_tasks/yields/

    Args:
        name (str): the dataset name.
        path (str, optional):
            The path to save the data file, defaults to './data'
        label_name (str, optional):
            For multi-label dataset, specify the label name, defaults to None
        print_stats (bool, optional):
            Whether to print basic statistics of the dataset, defaults to False
        convert_format (str, optional):
            Automatic conversion of SMILES to other molecular formats in MolConvert class. Stored as separate column in dataframe, defaults to None
    """

    def __init__(
        self,
        name,
        path="./data",
        label_name=None,
        print_stats=False,
        convert_format=None,
    ):
        """Create Yields (Reaction Yields Prediction) dataloader object."""
        super().__init__(
            name,
            path,
            label_name,
            print_stats,
            dataset_names=dataset_names["Yields"],
            convert_format=convert_format,
        )
        self.entity1_name = "Reaction"
        if print_stats:
            self.print_stats()
        print("Done!", flush=True, file=sys.stderr)
