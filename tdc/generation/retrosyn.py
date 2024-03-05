# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")

from . import generation_dataset
from ..metadata import dataset_names
from ..utils import create_fold


class RetroSyn(generation_dataset.PairedDataLoader):
    """Data loader class accessing to retro-synthetic prediction task."""

    def __init__(
        self,
        name,
        path="./data",
        print_stats=False,
        input_name="product",
        output_name="reactant",
    ):
        """To create an data loader object for forward reaction prediction task. The goal is to predict
        the reaction products given a set of reactants

        Args:
            name (str): the name of the datset
            path (str, optional): the path to the saved data file.
            print_stats (bool, optional): whether to print the basic statistics
            input_name (str, optional): the name of the column containing input molecular data (product)
            output_name (str, optional): the name of the column containing output molecular data (reactant)
        """
        super().__init__(name, path, print_stats, input_name, output_name)

    def get_split(
        self,
        method="random",
        seed=42,
        frac=[0.7, 0.1, 0.2],
        include_reaction_type=False,
    ):
        """Return the data splitted as train, valid, test sets.

        Arguments:
            method (str): splitting schemes: random, scaffold
            seed (int): random seed, default 42
            frac (list of float): ratio of train/val/test split
            include_reaction_type (bool): whether or not to include reaction type in the split

        Returns:
            pandas DataFrame/dict: a dataframe of the dataset

        Raises:
            AttributeError: Use the correct split method as input (random, scaffold)
        """
        df = self.get_data(format="df")

        if include_reaction_type:
            from ..utils import get_reaction_type

            try:
                rt = get_reaction_type(self.name)
                df["reaction_type"] = rt
            except:
                raise ValueError(
                    "Reaction Type Unavailable for " + str(self.name) +
                    "! Please turn include_reaction_type to be false!")

        if method == "random":
            return create_fold(df, seed, frac)
        else:
            raise AttributeError("Please use the correct split method")
