# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import sys
import warnings

warnings.filterwarnings("ignore")

from . import single_pred_dataset
from ..utils import print_sys, fuzzy_search, property_dataset_load
from ..metadata import dataset_names


class ADME(single_pred_dataset.DataLoader):
    """Data loader class to load datasets in ADME task. More info: https://tdcommons.ai/single_pred_tasks/adme/

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
        """Create ADME dataloader object."""
        super().__init__(
            name,
            path,
            label_name,
            print_stats,
            dataset_names=dataset_names["ADME"],
            convert_format=convert_format,
        )

        self.name = fuzzy_search(self.name, dataset_names["ADME"])
        if self.name == "ppbr_az":
            import pandas as pd
            import os

            self.ppbr_df = pd.read_csv(os.path.join(self.path,
                                                    self.name + ".tab"),
                                       sep="\t")
            df = self.ppbr_df[self.ppbr_df.Species == "Homo sapiens"]
            self.entity1 = df.Drug.values
            self.y = df.Y.values
            self.entity1_idx = df.Drug_ID.values

        if print_stats:
            self.print_stats()
        print("Done!", flush=True, file=sys.stderr)

    def get_approved_set(self):
        import pandas as pd

        if self.name not in ["pampa_ncats"]:
            raise ValueError(
                "This function is only available for PAMPA_NCATS dataset")
        entity1, y, entity1_idx = property_dataset_load("approved_pampa_ncats",
                                                        self.path, None,
                                                        dataset_names["ADME"])
        return pd.DataFrame({"Drug_ID": entity1_idx, "Drug": entity1, "Y": y})

    def get_other_species(self, species=None):

        if self.name not in ["ppbr_az"]:
            raise ValueError(
                "This function is only available for assays with species label, including PPBR"
            )

        if species == "all":
            return self.ppbr_df

        if species in self.ppbr_df.Species.unique():
            return self.ppbr_df[self.ppbr_df.Species == species].reset_index(
                drop=True)
        else:
            raise ValueError(
                "You can only specify the following set of species name: 'Canis lupus familiaris', 'Cavia porcellus', 'Homo sapiens', 'Mus musculus', 'Rattus norvegicus', 'all'"
            )

    def harmonize(self, mode=None):
        """Removing duplicated experimental readouts."""

        if mode not in ["max", "min", "remove_all"]:
            raise ValueError(
                "Please specify 'mode' of removal, currently supported 'max'/'min'/'remove_all'!"
            )

        if mode == "max":
            df_ = self.get_data()
            df = (df_.sort_values(
                "Y",
                ascending=True).drop_duplicates("Drug").reset_index(drop=True))

        elif mode == "min":
            df_ = self.get_data()
            df = (df_.sort_values(
                "Y",
                ascending=False).drop_duplicates("Drug").reset_index(drop=True))

        elif mode == "remove_all":
            df_ = self.get_data()
            df = df_.drop_duplicates("Drug", keep=False).reset_index(drop=True)

        self.entity1 = df.Drug.values
        self.y = df.Y.values
        self.entity1_idx = df.Drug_ID.values
        return df
