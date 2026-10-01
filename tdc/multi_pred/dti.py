# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")
import sys

from ..utils import print_sys
from . import bi_pred_dataset, multi_pred_dataset
from ..metadata import dataset_names


class DTI(bi_pred_dataset.DataLoader):
    """Data loader class to load datasets in Drug-Target Interaction Prediction task.
    More info: https://tdcommons.ai/multi_pred_tasks/dti/

    Regression task. Given the target amino acid sequence/compound SMILES string, predict their binding affinity.


    Args:
        name (str): the dataset name.
        path (str, optional):
            The path to save the data file, defaults to './data'
        label_name (str, optional):
            For multi-label dataset, specify the label name, defaults to None
        print_stats (bool, optional):
            Whether to print basic statistics of the dataset, defaults to False


    """

    def __init__(self,
                 name,
                 path="./data",
                 label_name=None,
                 print_stats=False):
        """Create Drug-Target Interaction Prediction dataloader object"""
        super().__init__(name,
                         path,
                         label_name,
                         print_stats,
                         dataset_names=dataset_names["DTI"])
        self.entity1_name = "Drug"
        self.entity2_name = "Target"
        self.two_types = True

        if print_stats:
            self.print_stats()

        print("Done!", flush=True, file=sys.stderr)

    def harmonize_affinities(self, mode=None):
        """Removing duplicated drug-target pairs with different binding affinities."""

        if mode not in ["mean", "max_affinity"]:
            raise ValueError(
                "Please specify 'mode' of removal, currently supported 'mean'/'max_affinity'!"
            )

        if mode == "max_affinity":
            if self.log_flag:
                print_sys(
                    "The scale is converted to log scale, so we will take the maximum!"
                )
                y_agg = "max"
            else:
                print_sys(
                    "The scale is in original affinity scale, so we will take the minimum!"
                )
                y_agg = "min"
        elif mode == "mean":
            y_agg = "mean"

        # Group by the drug-target pair rather than by auxiliary IDs: rows
        # with a missing Drug_ID or Target_ID are still valid pairs, but
        # pandas groupby drops NaN group keys by default.
        agg_map = {"Drug_ID": "first", "Target_ID": "first", "Y": y_agg}
        df = self.get_data().groupby(["Drug", "Target"],
                                     as_index=False).agg(agg_map)
        df = df[["Drug_ID", "Drug", "Target_ID", "Target", "Y"]]

        self.entity1_idx = df.Drug_ID.values
        self.entity2_idx = df.Target_ID.values

        self.entity1 = df.Drug.values
        self.entity2 = df.Target.values
        self.y = df.Y.values
        print_sys("The original data has been updated!")
        return df
