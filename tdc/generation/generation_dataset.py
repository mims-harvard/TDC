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
    distribution_dataset_load,
    generation_paired_dataset_load,
    three_dim_dataset_load,
    print_sys,
)
from ..utils import create_fold


class DataLoader(base_dataset.DataLoader):
    """A base dataset loader class.

    Attributes:
        dataset_names (str): name of the dataset.
        name (str): The name fo the dataset.
        path (str): the path to save the data file.
        smiles_lst (list): a list of smiles strings as training data for distribution learning.
    """

    def __init__(self, name, path, print_stats, column_name):
        """To create a base dataloader object that each generation task can inherit from.

        Args:
            name (str): the name of the dataset.
            path (str): the path to save the data file.
            print_stats (bool): whether to print the basic statistics of the dataset.
            column_name (str): The name of the column containing smiles strings.
        """
        from ..metadata import single_molecule_dataset_names

        self.smiles_lst = distribution_dataset_load(
            name, path, single_molecule_dataset_names, column_name=column_name)
        ### including fuzzy-search
        self.name = name
        self.path = path
        self.dataset_names = single_molecule_dataset_names
        if print_stats:
            self.print_stats()
        print_sys("Done!")

    def print_stats(self):
        """Print the basic statistics of the dataset."""
        print(
            "There are " + str(len(self.smiles_lst)) + " molecules ",
            flush=True,
            file=sys.stderr,
        )

    def get_data(self, format="df"):
        """Return the data from the whole dataset.

        Args:
            format (str, optional): the desired format for molecular data.

        Returns:
            pandas DataFrame/dict: a dataframe of the dataset/a distionary for information

        Raises:
            AttributeError: Use the correct format as input (df, dict)
        """
        if format == "df":
            return pd.DataFrame({"smiles": self.smiles_lst})
        elif format == "dict":
            return {"smiles": self.smiles_lst}
        else:
            raise AttributeError("Please use the correct format input")

    def get_split(self, method="random", seed=42, frac=[0.7, 0.1, 0.2]):
        """Return the data splitted as train, valid, test sets.

        Arguments:
            method (str): splitting schemes: random, scaffold
            seed (int): random seed, default 42
            frac (list of float): ratio of train/val/test split

        Returns:
            pandas DataFrame/dict: a dataframe of the dataset

        Raises:
            AttributeError: Use the correct split method as input (random, scaffold)
        """
        df = self.get_data(format="df")

        if method == "random":
            return create_fold(df, seed, frac)
        else:
            raise AttributeError("Please use the correct split method")


class PairedDataLoader(base_dataset.DataLoader):
    """A basic class for generation of biomedical entities conditioned on other entities, such as reaction prediction.

    Attributes:
        dataset_names (str): the name fo the dataset.
        name (str): the name of the dataset.
        path (str): the path to save the data file.
    """

    def __init__(self, name, path, print_stats, input_name, output_name):
        """To create a object for paired biomedical entities generation.

        Arguments:
            name (str): fuzzy name of the generation dataset. e.g., uspto50k, qed, drd, ...
            path (str): directory path that stores the dataset, e.g., ./data
            print_stats (bool): whether print the stats.
            input_name (str): The column name of input biomedical entities.
            output_name (str): The column name of output biomedical entities.

        """
        from ..metadata import paired_dataset_names

        self.input_smiles_lst, self.output_smiles_lst = generation_paired_dataset_load(
            name, path, paired_dataset_names, input_name,
            output_name)  ### including fuzzy-search
        self.name = name
        self.path = path
        self.dataset_names = paired_dataset_names
        if print_stats:
            self.print_stats()
        print_sys("Done!")

    def print_stats(self):
        """Print the statistics of the dataset."""
        print(
            "There are " + str(len(self.input_smiles_lst)) + " paired samples",
            flush=True,
            file=sys.stderr,
        )

    def get_data(self, format="df"):
        """Return the data from the whole dataset.

        Args:
            format (str, optional): the desired format for molecular data.

        Returns:
            pandas DataFrame/dict: a dataframe of the dataset/a distionary for information

        Raises:
            AttributeError: Use the correct format as input (df, dict)
        """
        if format == "df":
            return pd.DataFrame({
                "input": self.input_smiles_lst,
                "output": self.output_smiles_lst
            })
        elif format == "dict":
            return {
                "input": self.input_smiles_lst,
                "output": self.output_smiles_lst
            }
        else:
            raise AttributeError("Please use the correct format input")

    def get_split(self, method="random", seed=42, frac=[0.7, 0.1, 0.2]):
        """Return the data splitted as train, valid, test sets.

        Arguments:
            method (str): splitting schemes: random, scaffold
            seed (int): random seed, default 42
            frac (list of float): ratio of train/val/test split

        Returns:
            pandas DataFrame/dict: a dataframe of the dataset

        Raises:
            AttributeError: Use the correct split method as input (random, scaffold)
        """

        df = self.get_data(format="df")

        if method == "random":
            return create_fold(df, seed, frac)
        else:
            raise AttributeError("Please use the correct split method")


class DataLoader3D(base_dataset.DataLoader):
    """A basic class for generation of 3D biomedical entities. (under construction)

    Attributes:
        df (str): the dataset in pandas DataFrame format.
        name (str): the name of the dataset.
        path (str): the path to save the data file.
    """

    ### locally, unzip a folder, with the main file the dataframe with SMILES, Mol Object for various kinds of entities.
    ### also, for each column, contains a sdf file.

    def __init__(self, name, path, print_stats, dataset_names, column_name):
        """To create an object for 3D biomedical entities generation.

        Args:
            name (str): the name of the dataset.
            path (str): the path to save the data file.
            print_stats (bool): whether to print the basic statistics of the dataset.
            column_name (str): The name of the column containing smiles strings.
        """
        self.df, self.path, self.name = three_dim_dataset_load(
            name, path, dataset_names)
        if print_stats:
            self.print_stats()
        print_sys("Done!")

    def print_stats(self):
        """Print the basic statistics of the dataset."""
        print(
            "There are " + str(len(self.df)) + " data points ",
            flush=True,
            file=sys.stderr,
        )

    def get_data(self, format="df", more_features="None"):
        """Return the data from the whole dataset.

        Args:
            format (str, optional): the desired format for molecular data.
            more_features (str, optional): 3D feature format, choose from [Graph3D, Coulumb]

        Returns:
            pandas DataFrame/dict: a dataframe of the dataset/a distionary for information

        Raises:
            AttributeError: Use the correct format as input (df, dict)
            ImportError: Please install rdkit by 'conda install -c conda-forge rdkit'

        """
        if more_features in ["None", "SMILES"]:
            pass
        elif more_features in ["Graph3D", "Coulumb",
                               "SELFIES"]:  # why SELFIES here?
            try:
                from rdkit.Chem.PandasTools import LoadSDF
                from rdkit import rdBase

                rdBase.DisableLog("rdApp.error")
            except:
                raise ImportError(
                    "Please install rdkit by 'conda install -c conda-forge rdkit'! "
                )

            from ..chem_utils import MolConvert
            from ..metadata import sdf_file_names

            convert = MolConvert(src="SDF", dst=more_features)
            for i in sdf_file_names[self.name]:
                self.df[i + "_" + more_features] = convert(self.path + i +
                                                           ".sdf")

        if format == "df":
            return self.df
        else:
            raise AttributeError("Please use the correct format input")

    def get_split(self, method="random", seed=42, frac=[0.7, 0.1, 0.2]):
        """Return the data splitted as train, valid, test sets.

        Arguments:
            method (str): splitting schemes: random, scaffold
            seed (int): random seed, default 42
            frac (list of float): ratio of train/val/test split

        Returns:
            pandas DataFrame/dict: a dataframe of the dataset

        Raises:
            AttributeError: Use the correct split method as input (random, scaffold)
        """
        df = self.get_data(format="df")

        if method == "random":
            return create_fold(df, seed, frac)
        else:
            raise AttributeError("Please use the correct split method")
