# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")
import sys

from ..utils import print_sys
from . import bi_pred_dataset, multi_pred_dataset
from ..metadata import dataset_names


class TrialOutcome(multi_pred_dataset.DataLoader):
    """Data loader class to load datasets in clinical trial outcome Prediction task.
    More info: https://tdcommons.ai/multi_pred_tasks/trialoutcome/


    Task Description: Binary Classification.
                      Given the drug molecule, disease code (ICD) and trial protocol (eligibility criteria),
                      predict their trial approval rate.

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
        """Create Clinical Trial Outcome Prediction dataloader object"""
        super().__init__(name,
                         path,
                         print_stats,
                         dataset_names=dataset_names["TrialOutcome"])
        self.entity1_name = "drug_molecule"
        self.entity2_name = "disease_code"
        # self.entity3_name = "eligibility_criteria"

        if print_stats:
            self.print_stats()

        print("Done!", flush=True, file=sys.stderr)
