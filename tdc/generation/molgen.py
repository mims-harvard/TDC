# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")

from . import generation_dataset
from ..metadata import dataset_names


class MolGen(generation_dataset.DataLoader):
    """Data loader class accessing to molecular generation task (distribution learning)"""

    def __init__(self,
                 name,
                 path="./data",
                 print_stats=False,
                 column_name="smiles"):
        """To create an data loader object for molecular generation task. The goal is to generate diverse,
        novel molecules that has desirable chemical properties. One can combined with oracle functions.

        Args:
            name (str): the name of the datset
            path (str, optional): the path to the saved data file.
            print_stats (bool, optional): whether to print the basic statistics
            column_name (str, optional): the name of the column containing molecular data.
        """
        super().__init__(name, path, print_stats, column_name)
