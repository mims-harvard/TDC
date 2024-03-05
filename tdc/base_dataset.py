# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
"""
This file contains a base data loader object that specific one can inherit from. 
"""

import pandas as pd
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore")

from . import utils


class DataLoader:
    """base data loader class that contains functions shared by almost all data loader classes."""

    def __init__(self):
        """empty data loader class, to be overwritten"""
        pass

    def get_data(self, format="df"):
        """
        Arguments:
            format (str, optional): the dataset format

        Returns:
            pd.DataFrame/dict/np.array: when format is df/dict/DeepPurpose

        Raises:
            AttributeError: format not supported
        """
        if format == "df":
            return pd.DataFrame({
                self.entity1_name + "_ID": self.entity1_idx,
                self.entity1_name: self.entity1,
                "Y": self.y,
            })
        elif format == "dict":
            return {
                self.entity1_name + "_ID": self.entity1_idx,
                self.entity1_name: self.entity1,
                "Y": self.y,
            }
        elif format == "DeepPurpose":
            return self.entity1, self.y
        else:
            raise AttributeError("Please use the correct format input")

    def print_stats(self):
        """print statistics"""
        print(
            "There are " + str(len(np.unique(self.entity1))) + " unique " +
            self.entity1_name.lower() + "s",
            flush=True,
            file=sys.stderr,
        )

    def get_split(self, method="random", seed=42, frac=[0.7, 0.1, 0.2]):
        """
        split function, overwritten by single_pred/multi_pred/generation for more specific splits

        Arguments:
            method: splitting schemes
            seed: random seed
            frac: train/val/test split fractions

        Returns:
            dict: a dictionary of train/valid/test dataframes

        Raises:
            AttributeError: split method not supported
        """

        df = self.get_data(format="df")

        if method == "random":
            return utils.create_fold(df, seed, frac)
        elif method == "cold_" + self.entity1_name.lower():
            return utils.create_fold_setting_cold(df, seed, frac,
                                                  self.entity1_name)
        else:
            raise AttributeError("Please specify the correct splitting method")

    def label_distribution(self):
        """visualize distribution of labels"""
        utils.label_dist(self.y, self.name)

    def binarize(self, threshold=None, order="descending"):
        """binarize the labels

        Args:
            threshold (float, optional): the threshold to binarize the label.
            order (str, optional): the order of binarization, if ascending, flip 1 to larger values and vice versus for descending

        Returns:
            DataLoader: data loader class with updated label

        Raises:
            AttributeError: no threshold specified for binarization
        """
        if threshold is None:
            raise AttributeError(
                "Please specify the threshold to binarize the data by "
                "'binarize(threshold = N)'!")

        if len(np.unique(self.y)) == 2:
            print("The data is already binarized!", flush=True, file=sys.stderr)
        else:
            print(
                "Binariztion using threshold " + str(threshold) +
                ", default, we assume the smaller values are 1 "
                "and larger ones is 0, you can change the order "
                "by 'binarize(order = 'ascending')'",
                flush=True,
                file=sys.stderr,
            )
            if (np.unique(self.y).reshape(-1,).shape[0] < 2):
                raise AttributeError(
                    "Adjust your threshold, there is only one class.")
            self.y = utils.binarize(self.y, threshold, order)
        return self

    def __len__(self):
        """get number of data points

        Returns:
            int: number of data points
        """
        return len(self.get_data(format="df"))

    def convert_to_log(self, form="standard"):
        """convert labels to log-scale

        Args:
            form (str, optional): standard log-transformation or binding nM <-> p transformation.
        """
        print("To log space...", flush=True, file=sys.stderr)
        self.log_flag = True
        if form == "binding":
            self.y = utils.convert_to_log(self.y)
        elif form == "standard":
            self.sign = np.sign(self.y)
            self.y = self.sign * np.log(abs(self.y) + 1e-10)

    def convert_from_log(self, form="standard"):
        """convert labels from log-scale

        Args:
            form (str, optional): standard log-transformation or binding nM <-> p transformation.
        """
        print("Convert Back To Original space...", flush=True, file=sys.stderr)
        if form == "binding":
            self.y = utils.convert_back_log(self.y)
        elif form == "standard":
            self.y = self.sign * (np.exp(self.sign * self.y) - 1e-10)
        self.log_flag = False

    def get_label_meaning(self, output_format="dict"):
        """get the biomedical meaning of label

        Args:
            output_format (str, optional): dict/df/array for label

        Returns:
            dict/pd.DataFrame/np.array: when output_format is dict/df/array
        """
        return utils.get_label_map(
            self.name,
            self.path,
            self.target,
            file_format=self.file_format,
            output_format=output_format,
        )

    def balanced(self, oversample=False, seed=42):
        """balance the label neg-pos ratio

        Args:
            oversample (bool, optional): whether or not to oversample minority or subsample majority to match ratio
            seed (int, optional): random seed

        Returns:
            pd.DataFrame: the updated dataframe with balanced dataset

        Raises:
            AttributeError: alert to binarize the data first as continuous values cannot do balancing
        """
        if len(np.unique(self.y)) > 2:
            raise AttributeError(
                "You should binarize the data first by calling "
                "data.binarize(threshold)",
                flush=True,
                file=sys.stderr,
            )

        val = self.get_data()

        class_ = val.Y.value_counts().keys().values
        major_class = class_[0]
        minor_class = class_[1]

        if not oversample:
            print(
                " Subsample the majority class is used, if you want to do "
                "oversample the minority class, set 'balanced(oversample = True)'. ",
                flush=True,
                file=sys.stderr,
            )
            val = (pd.concat([
                val[val.Y == major_class].sample(
                    n=len(val[val.Y == minor_class]),
                    replace=False,
                    random_state=seed,
                ),
                val[val.Y == minor_class],
            ]).sample(frac=1, replace=False,
                      random_state=seed).reset_index(drop=True))
        else:
            print(" Oversample of minority class is used. ",
                  flush=True,
                  file=sys.stderr)
            val = (pd.concat([
                val[val.Y == minor_class].sample(
                    n=len(val[val.Y == major_class]),
                    replace=True,
                    random_state=seed,
                ),
                val[val.Y == major_class],
            ]).sample(frac=1, replace=False,
                      random_state=seed).reset_index(drop=True))
        return val
