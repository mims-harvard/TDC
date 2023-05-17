import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import os.path as osp
import h5py
import numpy as np
import warnings
from tqdm import tqdm

import torch 
import torch.nn.functional as F
from torch_geometric.data import Data


from .. import base_dataset
from ..utils.load import structure_based_protein_dataset_load
from ..utils.split import create_custom_split
from ..utils import fuzzy_search, print_sys
from ..metadata import dataset_names, structure_based_protein_dataset_names


class SBPP(base_dataset.DataLoader):

    """Data loader class accessing to structure-based protein property prediction task."""

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

        self.save = save


        protein, protein_property = structure_based_protein_dataset_load(
            name,
            path,
            structure_based_protein_dataset_names,
            return_pocket,
            threshold,
            remove_protein_Hs,
            remove_ligand_Hs,
            keep_het,
        )
        
        if save:
            np.savez(
                os.path.join(path, name + ".npz"),
                protein_amino_type=protein["amino_type"],
                protein_coord=protein["coord"],
                protein_atom_name=protein["atom_name"],
                protein_atom_amino_id=protein["atom_amino_id"],
                protein_property=protein_property["prop"]
            )
        
    
        self.protein = protein
        self.protein_property = protein_property
    
        self.name = name
        self.path = path
        self.return_pocket = return_pocket
        if print_stats:
            self.print_stats()
        print_sys("Done!")


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
            return {"protein": self.protein, "protein_property": self.protein_property}
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
        splitted_data = create_custom_split(self.name, data)
        
        if self.save:
            np.savez(
                os.path.join(self.path, self.name + "_train.npz"),
                protein_coord=splitted_data["train"]["protein_coord"],
                protein_amino_type=splitted_data["train"]["protein_amino_type"],
                protein_atom_name=splitted_data["train"]["protein_atom_name"],
                protein_atom_amino_id=splitted_data["train"]["protein_atom_amino_id"],
                protein_property=splitted_data["train"]["protein_property"]
            )
            np.savez(
                os.path.join(self.path, self.name + "_val.npz"),
                protein_coord=splitted_data["val"]["protein_coord"],
                protein_amino_type=splitted_data["val"]["protein_amino_type"],
                protein_atom_name=splitted_data["val"]["protein_atom_name"],
                protein_atom_amino_id=splitted_data["val"]["protein_atom_amino_id"],
                protein_property=splitted_data["val"]["protein_property"]
            )
            np.savez(
                os.path.join(self.path, self.name + "_test.npz"),
                protein_coord=splitted_data["test"]["protein_coord"],
                protein_amino_type=splitted_data["test"]["protein_amino_type"],
                protein_atom_name=splitted_data["test"]["protein_atom_name"],
                protein_atom_amino_id=splitted_data["test"]["protein_atom_amino_id"],
                protein_property=splitted_data["test"]["protein_property"]
            )
            
        return splitted_data



if __name__ == "__main__":
    dataset = SBPP(name='enzyme_catalysis', path='./data')
    a = dataset.get_split()
  