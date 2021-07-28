# -*- coding: utf-8 -*-
"""Docstring to be finished.
"""
# Author: TDC Team
# License: MIT

import sys
import warnings
warnings.filterwarnings("ignore")

from . import single_pred_dataset
from ..utils import print_sys, train_val_test_split
from ..metadata import dataset_names


class ADME(single_pred_dataset.DataLoader):

    """Summary
    """
    
    def __init__(self, name, path='./data', label_name=None, print_stats=False, convert_format=None):
        """Summary
        
        Args:
            name (TYPE): Description
            path (str, optional): Description
            label_name (None, optional): Description
            print_stats (bool, optional): Description
            convert_format (None, optional): Description
        """
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["ADME"], convert_format = convert_format)
        if print_stats:
            self.print_stats()
        print('Done!', flush = True, file = sys.stderr)