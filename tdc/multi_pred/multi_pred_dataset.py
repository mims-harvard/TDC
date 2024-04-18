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
    multi_dataset_load,
    create_fold,
    create_fold_setting_cold,
    create_combination_split,
    print_sys,
)


class DataLoader(base_dataset.DataLoader):
    """A base data loader class that each multi-instance prediction task dataloader class can inherit from.

    Attributes: TODO

    """

    def __init__(self, name, path, print_stats, dataset_names):
        """create dataloader object

        Args:
                name (str): name of dataloader
                path (str): the path where data is saved
                label_name (str): name of label
                print_stats (bool): whether to print statistics of dataset
                dataset_names (str): A list of dataset names available for a task
        """
        if name.lower() in dataset2target_lists.keys():
            if label_name is None:
                raise ValueError(
                    "Please select a label name. You can use tdc.utils.retrieve_label_name_list('"
                    + name.lower() +
                    "') to retrieve all available label names.")

        df = multi_dataset_load(name, path, dataset_names)

        self.df = df
        self.name = name
        self.path = path

    def get_data(self, format="df"):
        """generate data in some format, e.g., pandas.DataFrame

        Args:
            format (str, optional):
                format of data, the default value is 'df' (DataFrame)

        Returns:
            pandas DataFrame/dict: a dataframe of a dataset/a dictionary for key information in the dataset

        Raises:
            AttributeError: Use the correct format input (df, dict, DeepPurpose)
        """
        if format == "df":
            return self.df
        elif format == "dict":
            return dict(self.df)
        else:
            raise AttributeError("Please use the correct format input")

    def print_stats(self):
        """print the statistics of the dataset"""
        print_sys("--- Dataset Statistics ---")
        print(str(len(self.df)) + " data points.", flush=True, file=sys.stderr)
        print_sys("--------------------------")

    def get_split(self,
                  method="random",
                  seed=42,
                  frac=[0.7, 0.1, 0.2],
                  column_name=None,
                  data_ready=False):
        """split dataset into train/validation/test.

        Args:
            method (str, optional):
                split method, the default value is 'random'
            seed (int, optional):
                random seed, defaults to '42'
            frac (list, optional):
                train/val/test split fractions, defaults to '[0.7, 0.1, 0.2]'
            column_name (None, optional): Description

        Returns:
            dict: a dictionary with three keys ('train', 'valid', 'test'), each value is a pandas dataframe object of the splitted dataset

        Raises:
            AttributeError: the input split method is not available.

        """
        df = self.get_data(format="df") if not data_ready else self.df

        if method == "random":
            return create_fold(df, seed, frac)

        elif method == "cold_split":
            if isinstance(column_name, str):
                column_name = [column_name]
            if (column_name is None) or (not all(
                [x in df.columns.values for x in column_name])):
                raise AttributeError(
                    "For cold_split, please provide one or multiple column names that are contained in the dataframe."
                )
            return create_fold_setting_cold(df, seed, frac, column_name)

        elif method == "combination":
            return create_combination_split(df, seed, frac)
        else:
            raise AttributeError(
                "Please select a splitting strategy from random, cold_split, or combination."
            )
