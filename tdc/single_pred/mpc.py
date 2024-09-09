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
from ..utils import create_fold_setting_cold, create_scaffold_split
from ..metadata import dataset_names


class MPC(single_pred_dataset.DataLoader):

    def __init__(self, name, path="./data"):
        self.name = name
        self.data = None
        self.is_molace = False

    def get_from_gh(self, link):
        import pandas as pd
        import requests
        import io

        data = requests.get(link)
        try:
            data.raise_for_status()
        except:
            raise Exception(
                "invalid link provided. choose a link for datasets in https://github.com/bidd-group/MPCD"
            )
        self.data = pd.read_csv(io.StringIO(data.text))
        return self.data

    def get_data(self, link=None, get_from_gh=True, **kwargs):
        link = link if link is not None else self.name
        if get_from_gh:
            return self.get_from_gh(link)
        # support direct interfface with MoleculeACE API as well
        from MoleculeACE import Data, Descriptors
        self.molace = True
        try:
            self.data = Data(self.name)
            self.data(Descriptors.SMILES)
        except:
            raise Exception(
                "could not find dataset {}. For list of MoleculeAce datasets see https://github.com/bidd-group/MPCD/tree/main?tab=readme-ov-file#overview-of-the-mpc-benchmark-datasets"
                .format(self.name))
        return self.data

    def get_split(self, method="scaffold", seed=42, frac=[0.7, 0.1, 0.2]):
        d = self.get_data()
        if not self.is_molace:
            if method == "scaffold":
                return create_scaffold_split(d,
                                             seed=seed,
                                             frac=frac,
                                             entity="SMILES")
            elif method == "cold":
                return create_fold_setting_cold(d,
                                                seed=seed,
                                                frac=frac,
                                                entities="SMILES")
            raise Exception(
                "only scaffold or cold splits supported for the MPC task")
        train = pd.concat([d.x_train, d.y_train], axis=1)
        test = pd.concat([d.x_test, d.y_test], axis=1)
        return {
            "train": train,
            "test": test,
        }
