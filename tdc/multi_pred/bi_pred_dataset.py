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
    fuzzy_search,
    interaction_dataset_load,
    label_transform,
    NegSample,
    install,
    create_fold,
    create_fold_setting_cold,
    create_combination_split,
    create_fold_time,
    print_sys,
)


class DataLoader(base_dataset.DataLoader):
    """A base data loader class that each bi-instance prediction task dataloader class can inherit from.

    Attributes: TODO

    """

    def __init__(self,
                 name,
                 path,
                 label_name,
                 print_stats,
                 dataset_names,
                 data_config=None):
        """Create a base dataloader object that each multi-instance prediction task dataloader class can inherit from.

        Args:
            name (str): name of dataloader
            path (str): the path where data is saved
            label_name (str): name of label
            print_stats (bool): whether to print statistics of dataset
            dataset_names (str): A list of dataset names available for a task

        Raises:
            ValueError: label name is not available
        """
        if name.lower() in dataset2target_lists.keys():
            # print_sys("Tip: Use tdc.utils.retrieve_label_name_list(
            # '" + name.lower() + "') to retrieve all available label names.")
            if label_name is None:
                raise ValueError(
                    "Please select a label name. "
                    "You can use tdc.utils.retrieve_label_name_list('" +
                    name.lower() + "') to retrieve all available label names.")

        name = fuzzy_search(name, dataset_names)
        if name == "bindingdb_patent":
            aux_column = "Year"
        else:
            aux_column = None

        (entity1, entity2, raw_y, entity1_idx, entity2_idx, aux_column_val, df,
         augment_df) = interaction_dataset_load(name,
                                                path,
                                                label_name,
                                                dataset_names,
                                                aux_column=aux_column,
                                                data_config=data_config)

        self.name = name
        self.entity1 = entity1
        self.entity2 = entity2
        self.raw_y = raw_y
        self.y = raw_y
        self.entity1_idx = entity1_idx
        self.entity2_idx = entity2_idx
        self.path = path
        self.file_format = "csv"
        self.label_name = label_name

        self.entity1_name = "Entity1"
        self.entity2_name = "Entity2"
        self.aux_column = aux_column
        self.aux_column_val = aux_column_val
        self.log_flag = False
        self.two_types = False
        self.augment_df = augment_df
        self.df = df if augment_df else None

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
            out = None
            if self.aux_column is None:
                out = pd.DataFrame({
                    self.entity1_name + "_ID": self.entity1_idx,
                    self.entity1_name: self.entity1,
                    self.entity2_name + "_ID": self.entity2_idx,
                    self.entity2_name: self.entity2,
                    "Y": self.y,
                })
            else:
                out = pd.DataFrame({
                    self.entity1_name + "_ID": self.entity1_idx,
                    self.entity1_name: self.entity1,
                    self.entity2_name + "_ID": self.entity2_idx,
                    self.entity2_name: self.entity2,
                    "Y": self.y,
                    self.aux_column: self.aux_column_val,
                })
            if not self.augment_df:
                return out
            for col in self.df.columns:
                out[col] = self.df[col].values
            return out

        elif format == "DeepPurpose":
            return self.entity1.values, self.entity2.values, self.y.values
        elif format == "dict":
            out = {
                self.entity1_name + "_ID": self.entity1_idx.values,
                self.entity1_name: self.entity1.values,
                self.entity2_name + "_ID": self.entity2_idx.values,
                self.entity2_name: self.entity2.values,
                "Y": self.y.values,
            }
            if not self.augment_df:
                return out
            for col in self.df.columns:
                out[col] = self.df[col].values
            return out
        else:
            raise AttributeError("Please use the correct format input")

    def print_stats(self):
        """print the statistics of the dataset"""
        print_sys("--- Dataset Statistics ---")
        try:
            x = np.unique(self.entity1)
        except:
            x = np.unique(self.entity1_idx)

        try:
            y = np.unique(self.entity2)
        except:
            y = np.unique(self.entity2_idx)

        print(
            str(len(x)) + " unique " + self.entity1_name.lower() + "s.",
            flush=True,
            file=sys.stderr,
        )
        print(
            str(len(y)) + " unique " + self.entity2_name.lower() + "s.",
            flush=True,
            file=sys.stderr,
        )
        print(
            str(len(self.y)) + " " + self.entity1_name.lower() + "-" +
            self.entity2_name.lower() + " pairs.",
            flush=True,
            file=sys.stderr,
        )
        print_sys("--------------------------")

    def get_split(
        self,
        method="random",
        seed=42,
        frac=[0.7, 0.1, 0.2],
        column_name=None,
        time_column=None,
    ):
        """split dataset into train/validation/test.

        Args:
            method (str, optional):
                split method, the default value is 'random'
            seed (int, optional):
                random seed, defaults to '42'
            frac (list, optional):
                train/val/test split fractions, defaults to '[0.7, 0.1, 0.2]'
            column_name (Optional[Union[str, List[str]]]): Optional column name(s) to
                split on for cold splits. Defaults to None.
            time_column (None, optional): Description

        Returns:
            dict: a dictionary with three keys ('train', 'valid', 'test'), each value
            is a pandas dataframe object of the splitted dataset.

        Raises:
            AttributeError: the input split method is not available.

        """
        df = self.get_data(format="df")

        if isinstance(column_name, str):
            column_name = [column_name]

        if method == "random":
            return create_fold(df, seed, frac)
        elif method == "cold_" + self.entity1_name.lower():
            return create_fold_setting_cold(df, seed, frac, self.entity1_name)
        elif method == "cold_" + self.entity2_name.lower():
            return create_fold_setting_cold(df, seed, frac, self.entity2_name)
        elif method == "cold_split":
            if column_name is None or not all(
                    list(map(lambda x: x in df.columns.values, column_name))):
                raise AttributeError(
                    "For cold_split, please provide one or multiple column names "
                    "that are contained in the dataframe.")
            return create_fold_setting_cold(df, seed, frac, column_name)
        elif method == "combination":
            return create_combination_split(df, seed, frac)
        elif method == "time":
            if time_column is None:
                raise ValueError(
                    "Please specify the column that has the time variable using time_column."
                )
            return create_fold_time(df, frac, time_column)

        else:
            raise AttributeError(
                "Please select method from random, time, combination or cold_split."
            )

    def neg_sample(self, frac=1):
        """negative sampling

        Args:
            frac (int, optional): the ratio between negative and positive samples.

        Returns:
            DataLoader, the class itself.
        """
        df = NegSample(
            df=self.get_data(format="df"),
            column_names=[
                self.entity1_name + "_ID",
                self.entity1_name,
                self.entity2_name + "_ID",
                self.entity2_name,
            ],
            frac=frac,
            two_types=self.two_types,
        )
        self.entity1_idx = df[self.entity1_name + "_ID"]
        self.entity2_idx = df[self.entity2_name + "_ID"]
        self.entity1 = df[self.entity1_name]
        self.entity2 = df[self.entity2_name]
        self.y = df["Y"]
        self.raw_y = self.y
        return self

    def to_graph(
        self,
        threshold=None,
        format="edge_list",
        split=True,
        frac=[0.7, 0.1, 0.2],
        seed=42,
        order="descending",
    ):
        """Summary TODO

        Args:
            threshold (float, optional): threshold to binarize the data.
            format (str, optional): format of data, defaults to 'edge_list'
            split (bool, optional): if we need to split data into train/valid/test.
            frac (list, optional):  train/val/test split fractions, defaults to '[0.7, 0.1, 0.2]'
            seed (int, optional):  random seed, defaults to '42'
            order (str, optional): order of label transform

        Returns:
            dict: a dictionary for key information in the dataset

        Raises:
            AttributeError: the threshold is not available.
            ImportError: install the required package
        """
        df = self.get_data(format="df")

        if len(np.unique(self.raw_y)) > 2:
            print(
                "The dataset label consists of affinity scores. "
                "Binarization using threshold " + str(threshold) +
                " is conducted to construct the positive edges in the network. "
                "Adjust the threshold by to_graph(threshold = X)",
                flush=True,
                file=sys.stderr,
            )
            if threshold is None:
                raise AttributeError(
                    "Please specify the threshold to binarize the data by "
                    "'to_graph(threshold = N)'!")
            df["label_binary"] = label_transform(self.raw_y,
                                                 True,
                                                 threshold,
                                                 False,
                                                 verbose=False,
                                                 order=order)
        else:
            # already binary
            df["label_binary"] = df["Y"]

        df[self.entity1_name + "_ID"] = df[self.entity1_name +
                                           "_ID"].astype(str)
        df[self.entity2_name + "_ID"] = df[self.entity2_name +
                                           "_ID"].astype(str)
        df_pos = df[df.label_binary == 1]
        df_neg = df[df.label_binary == 0]

        return_dict = {}

        pos_edges = df_pos[[
            self.entity1_name + "_ID", self.entity2_name + "_ID"
        ]].values
        neg_edges = df_neg[[
            self.entity1_name + "_ID", self.entity2_name + "_ID"
        ]].values
        edges = df[[self.entity1_name + "_ID",
                    self.entity2_name + "_ID"]].values

        if format == "edge_list":
            return_dict["edge_list"] = pos_edges
            return_dict["neg_edges"] = neg_edges
        elif format == "dgl":
            try:
                import dgl
            except:
                install("dgl")
                import dgl
            unique_entities = np.unique(pos_edges.T.flatten()).tolist()
            index = list(range(len(unique_entities)))
            dict_ = dict(zip(unique_entities, index))
            edge_list1 = np.array([dict_[i] for i in pos_edges.T[0]])
            edge_list2 = np.array([dict_[i] for i in pos_edges.T[1]])
            return_dict["dgl_graph"] = dgl.DGLGraph((edge_list1, edge_list2))
            return_dict["index_to_entities"] = dict_

        elif format == "pyg":
            try:
                import torch
                from torch_geometric.data import Data
            except:
                raise ImportError(
                    "Please see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html to install pytorch geometric!"
                )

            unique_entities = np.unique(pos_edges.T.flatten()).tolist()
            index = list(range(len(unique_entities)))
            dict_ = dict(zip(unique_entities, index))
            edge_list1 = np.array([dict_[i] for i in pos_edges.T[0]])
            edge_list2 = np.array([dict_[i] for i in pos_edges.T[1]])

            edge_index = torch.tensor([edge_list1, edge_list2],
                                      dtype=torch.long)
            x = torch.tensor(np.array(index), dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            return_dict["pyg_graph"] = data
            return_dict["index_to_entities"] = dict_

        elif format == "df":
            return_dict["df"] = df

        if split:
            return_dict["split"] = create_fold(df, seed, frac)

        return return_dict
