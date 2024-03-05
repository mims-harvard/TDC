# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import base_dataset
from ..utils import bi_distribution_dataset_load, print_sys
from ..utils import create_combination_generation_split


class DataLoader(base_dataset.DataLoader):
    """A base dataset loader class.

    Attributes:
        dataset_names (str): name of the dataset.
        name (str): The name fo the dataset.
        path (str): the path to save the data file.
        smiles_lst (list): a list of smiles strings as training data for distribution learning.
    """

    def __init__(
        self,
        name,
        path,
        print_stats,
        return_pocket=False,
        threshold=15,
        remove_Hs=True,
        keep_het=False,
        allowed_atom_list=[
            "C", "N", "O", "S", "H", "B", "Br", "Cl", "P", "I", "F"
        ],
    ):
        """To create a base dataloader object that each generation task can inherit from.

        Args:
            name (str): the name of the dataset.
            path (str): the path to save the data file.
            print_stats (bool): whether to print the basic statistics of the dataset.
            column_name (str): The name of the column containing smiles strings.
        """
        from ..metadata import multiple_molecule_dataset_names

        protein, ligand = bi_distribution_dataset_load(
            name,
            path,
            multiple_molecule_dataset_names,
            return_pocket,
            threshold,
            remove_Hs,
            keep_het,
            allowed_atom_list,
        )

        self.ligand = ligand
        self.protein = protein

        ### including fuzzy-search
        self.name = name
        self.path = path
        self.dataset_names = multiple_molecule_dataset_names
        self.return_pocket = return_pocket
        self.remove_Hs = remove_Hs
        self.keep_het = keep_het
        self.allowed_atom_list = allowed_atom_list
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
        if format == "dict":
            return {"protein": self.protein, "ligand": self.ligand}
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
        data = self.get_data(format="dict")
        protein, ligand = data["protein"], data["ligand"]

        if method == "random":
            return create_combination_generation_split(protein, ligand, seed,
                                                       frac)
        else:
            raise AttributeError("Please use the correct split method")
