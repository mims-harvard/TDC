# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")
import sys

from ..utils import print_sys
from . import bi_pred_dataset, multi_pred_dataset
from ..metadata import dataset_names


class TestMultiPred(bi_pred_dataset.DataLoader):
    """Summary

    Attributes:
        entity1_name (str): Description
        entity2_name (str): Description
        two_types (bool): Description
    """

    def __init__(self, name, path="./data", label_name=None, print_stats=False):
        """Summary

        Args:
            name (TYPE): Description
            path (str, optional): Description
            label_name (None, optional): Description
            print_stats (bool, optional): Description
        """
        super().__init__(
            name,
            path,
            label_name,
            print_stats,
            dataset_names=dataset_names["test_multi_pred"],
        )
        self.entity1_name = "Antibody"
        self.entity2_name = "Antigen"
        self.two_types = True

        if print_stats:
            self.print_stats()

        print("Done!", flush=True, file=sys.stderr)
