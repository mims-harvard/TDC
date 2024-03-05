# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT"

import warnings

warnings.filterwarnings("ignore")

from . import generation_dataset
from ..metadata import dataset_names


class Reaction(generation_dataset.PairedDataLoader):
    """Data loader class accessing to forward reaction prediction task."""

    def __init__(
        self,
        name,
        path="./data",
        print_stats=False,
        input_name="reactant",
        output_name="product",
    ):
        """To create an data loader object for forward reaction prediction task. The goal is to predict
        the reaction products given a set of reactants

        Args:
            name (str): the name of the datset
            path (str, optional): the path to the saved data file.
            print_stats (bool, optional): whether to print the basic statistics
            input_name (str, optional): the name of the column containing input molecular data (reactant)
            output_name (str, optional): the name of the column containing output molecular data (product)
        """
        super().__init__(name, path, print_stats, input_name, output_name)
