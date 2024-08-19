# Molecular Property Cliff Task
# https://github.com/bidd-group/MPCD

# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import pandas as pd
import sys
import warnings

warnings.filterwarnings("ignore")

from . import single_pred_dataset
from ..utils import print_sys, fuzzy_search, property_dataset_load
from ..metadata import dataset_names


class MPC(single_pred_dataset.DataLoader):

    def __init__(self, name, path="./data"):
        self.name = name
        self.data = None

    def get_data(self):
        from MoleculeACE import Data, Descriptors  #TODO: support non-MoleculeACE
        try:
            self.data = Data(self.name)
            self.data(Descriptors.SMILES)
        except:
            raise Exception(
                "could not find dataset {}. For list of MoleculeAce datasets see https://github.com/bidd-group/MPCD/tree/main?tab=readme-ov-file#overview-of-the-mpc-benchmark-datasets"
                .format(self.name))
        return self.data

    def get_split(self):
        d = self.get_data()
        train = pd.concat([d.x_train, d.y_train], axis=1)
        test = pd.concat([d.x_test, d.y_test], axis=1)
        return {
            "train": train,
            "test": test,
        }
