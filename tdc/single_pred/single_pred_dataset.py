# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import base_dataset
from ..utils import (
    dataset2target_lists,
    property_dataset_load,
    create_fold,
    create_fold_setting_cold,
    create_scaffold_split,
    print_sys,
)


class DataLoader(base_dataset.DataLoader):
    """A base data loader class.

    Args:
        name (str): the dataset name.
        path (str): The path to save the data file
        label_name (str): For multi-label dataset, specify the label name
        print_stats (bool): Whether to print basic statistics of the dataset
        dataset_names (list): A list of dataset names available for a task
        convert_format (str): Automatic conversion of SMILES to other molecular formats in MolConvert class. Stored as separate column in dataframe

    Attributes:
        convert_format (str): conversion format of an entity
        convert_result (list): a placeholder for a list of conversion outputs
        entity1 (Pandas Series): a list of the single entites
        entity1_idx (Pandas Series): a list of the single entites index
        entity1_name (Pandas Series): a list of the single entites names
        file_format (str): the format of the downloaded dataset
        label_name (str): for multi-label dataset, the label name of interest
        name (str): dataset name
        path (str): path to save and retrieve the dataset
        y (Pandas Series): a list of the single entities label
    """

    def __init__(
        self,
        name,
        path,
        label_name,
        print_stats,
        dataset_names,
        convert_format,
        raw_format="SMILES",
    ):
        """Create a base dataloader object that each single instance prediction task dataloader class can inherit from.

        Raises:
            ValueError: for a dataset with multiple labels, specify the label. Use tdc.utils.retrieve_label_name_list to see the available label names

        """
        if name.lower() in dataset2target_lists.keys():
            if label_name is None:
                raise ValueError(
                    "Please select a label name. You can use tdc.utils.retrieve_label_name_list('"
                    + name.lower() +
                    "') to retrieve all available label names.")

        entity1, y, entity1_idx = property_dataset_load(name, path, label_name,
                                                        dataset_names)

        self.entity1 = entity1
        self.y = y
        self.entity1_idx = entity1_idx
        self.name = name
        self.entity1_name = "Drug"
        self.path = path
        self.file_format = "csv"
        self.label_name = label_name
        self.convert_format = convert_format
        self.convert_result = None
        self.raw_format = raw_format  ### 'SMILES' for most data, 'Raw3D' for QM9, ...

    def get_data(self, format="df"):
        """
        Arguments:
            format (str, optional): the returning dataset format, defaults to 'df'

        Returns:
            pandas DataFrame/dict: a dataframe of a dataset/a dictionary for key information in the dataset

        Raises:
            AttributeError: Use the correct format input (df, dict, DeepPurpose)
        """

        if (self.convert_format is not None) and (self.convert_result is None):
            from ..chem_utils import MolConvert

            converter = MolConvert(src=self.raw_format, dst=self.convert_format)
            convert_result = converter(self.entity1.values)
            self.convert_result = [i for i in convert_result]

        if format == "df":
            if self.convert_format is not None:
                return pd.DataFrame({
                    self.entity1_name + "_ID":
                        self.entity1_idx,
                    self.entity1_name:
                        self.entity1,
                    self.entity1_name + "_" + self.convert_format:
                        self.convert_result,
                    "Y":
                        self.y,
                })
            else:
                return pd.DataFrame({
                    self.entity1_name + "_ID": self.entity1_idx,
                    self.entity1_name: self.entity1,
                    "Y": self.y,
                })

        elif format == "dict":
            if self.convert_format is not None:
                return {
                    self.entity1_name + "_ID":
                        self.entity1_idx.values,
                    self.entity1_name:
                        self.entity1.values,
                    self.entity1_name + "_" + self.convert_format:
                        self.convert_result,
                    "Y":
                        self.y.values,
                }
            else:
                return {
                    self.entity1_name + "_ID": self.entity1_idx.values,
                    self.entity1_name: self.entity1.values,
                    "Y": self.y.values,
                }
        elif format == "DeepPurpose":
            return self.entity1.values, self.y.values
        else:
            raise AttributeError("Please use the correct format input")

    def get_split(self, method="random", seed=42, frac=[0.7, 0.1, 0.2]):
        """
        Arguments:
            method: splitting schemes, choose from random, cold_{entity}, scaffold, defaults to 'random'
            seed: the random seed for splitting dataset, defaults to '42'
            frac: train/val/test split fractions, defaults to '[0.7, 0.1, 0.2]'

        Returns:
            dict: a dictionary with three keys ('train', 'valid', 'test'), each value is a pandas dataframe object of the splitted dataset

        Raises:
            AttributeError: the input split method is not available.
        """

        df = self.get_data(format="df")

        if method == "random":
            split = create_fold(df, seed, frac)
        elif method == "cold_" + self.entity1_name.lower():
            split = create_fold_setting_cold(df, seed, frac, self.entity1_name)
        elif method == "scaffold":
            split = create_scaffold_split(df, seed, frac, self.entity1_name)
        else:
            raise AttributeError("Please specify the correct splitting method")
        self.split = split
        return split

    def print_stats(self):
        """Print basic data statistics."""
        print_sys("--- Dataset Statistics ---")
        try:
            x = np.unique(self.entity1)
        except:
            x = np.unique(self.entity1_idx)

        print(
            str(len(x)) + " unique " + self.entity1_name.lower() + "s.",
            flush=True,
            file=sys.stderr,
        )
        print_sys("--------------------------")
