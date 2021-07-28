"""Summary
"""
import warnings
warnings.filterwarnings("ignore")
import sys

from ..utils import print_sys
from . import bi_pred_dataset, multi_pred_dataset
from ..metadata import dataset_names

class DrugSyn(multi_pred_dataset.DataLoader):

    """Summary
    """
    
    def __init__(self, name, path='./data', print_stats=False):
        """Summary
        
        Args:
            name (TYPE): Description
            path (str, optional): Description
            print_stats (bool, optional): Description
        """
        super().__init__(name, path, print_stats,
                         dataset_names=dataset_names["DrugSyn"])

        if print_stats:
            self.print_stats()

        print('Done!', flush=True, file=sys.stderr)