# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT"

import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import base_dataset
from ..utils import bi_distribution_dataset_load, print_sys
from ..utils import create_combination_generation_split
from ..metadata import dataset_names


class SBDD(base_dataset.DataLoader):
    """Data loader class accessing to structure-based drug design task."""

    def __init__(
        self,
        name,
        path="./data",
        print_stats=False,
        return_pocket=False,
        threshold=15,
        remove_protein_Hs=True,
        remove_ligand_Hs=True,
        keep_het=False,
        save=True,
    ):
        """To create a base dataloader object for structure-based drug design task.

        Args:
            name (str): the name of the dataset.
            path (str): the path to save the data file.
            print_stats (bool): whether to print the basic statistics of the dataset.
            return_pocket (bool): whether to return only protein pocket or full protein.
                threshold (int): only enabled when return_pocket is to True, if pockets are not provided in the raw data,
                                                 the threshold is used as a radius for a sphere around the ligand center to consider protein pocket.
                remove_protein_Hs (bool): whether to remove H atoms from proteins or not.
                remove_ligand_Hs (bool): whether to remove H atoms from ligands or not.
                keep_het (bool): whether to keep het atoms (e.g. cofactors) in protein.
                allowed_atom_list (list(str)): atom types allowed to include.
                save (bool): whether to save preprocessed data and splits.
        """
        from ..metadata import multiple_molecule_dataset_names

        try:
            import biopandas
        except:
            raise ImportError(
                "Please install biopandas by 'pip install biopandas'! ")
        protein, ligand = bi_distribution_dataset_load(
            name,
            path,
            multiple_molecule_dataset_names,
            return_pocket,
            threshold,
            remove_protein_Hs,
            remove_ligand_Hs,
            keep_het,
        )
        if save:
            np.savez(
                os.path.join(path, name + ".npz"),
                protein_coord=protein["coord"],
                protein_atom=protein["atom_type"],
                ligand_coord=ligand["coord"],
                ligand_atom=ligand["atom_type"],
            )
        self.save = save

        self.ligand = ligand
        self.protein = protein

        ### including fuzzy-search
        self.name = name
        self.path = path
        self.dataset_names = multiple_molecule_dataset_names
        self.return_pocket = return_pocket
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

    def get_data(self, format="dict"):
        """Return the data from the whole dataset.

        Args:
            format (str, optional): the desired format for molecular data.

        Returns:
            dict: a dict of protein, ligand information

        Raises:
            AttributeError: Use the correct format as input (dict)
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

        splitted_data = create_combination_generation_split(
            protein, ligand, seed, frac)

        if self.save:
            np.savez(
                os.path.join(self.path, self.name + "_train.npz"),
                protein_coord=splitted_data["train"]["protein_coord"],
                protein_atom=splitted_data["train"]["protein_atom_type"],
                ligand_coord=splitted_data["train"]["ligand_coord"],
                ligand_atom=splitted_data["train"]["ligand_atom_type"],
            )
            np.savez(
                os.path.join(self.path, self.name + "_valid.npz"),
                protein_coord=splitted_data["valid"]["protein_coord"],
                protein_atom=splitted_data["valid"]["protein_atom_type"],
                ligand_coord=splitted_data["valid"]["ligand_coord"],
                ligand_atom=splitted_data["valid"]["ligand_atom_type"],
            )
            np.savez(
                os.path.join(self.path, self.name + "_test.npz"),
                protein_coord=splitted_data["test"]["protein_coord"],
                protein_atom=splitted_data["test"]["protein_atom_type"],
                ligand_coord=splitted_data["test"]["ligand_coord"],
                ligand_atom=splitted_data["test"]["ligand_atom_type"],
            )

        if method == "random":
            return splitted_data
        else:
            raise AttributeError("Please use the correct split method")
