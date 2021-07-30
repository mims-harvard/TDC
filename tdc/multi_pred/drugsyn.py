"""Summary
"""
import warnings
warnings.filterwarnings("ignore")
import sys

from ..utils import print_sys
from . import bi_pred_dataset, multi_pred_dataset
from ..metadata import dataset_names

class DrugSyn(multi_pred_dataset.DataLoader):

    """Drug Synergy Prediction 
    Given the chemical structural information for the combining drugs and genomic features for a particular cell line, 
    our goal is to predict the drug synergy level or the drug combination sensitivity in that cell line.
    """
    
    def __init__(self, name, path='./data', print_stats=False):
        """initialize function of DrugSyn. 
        
        Args:
            name (str): can be 'OncoPolyPharmacology' or 'DrugComb'
            path (str, optional): the path it save, the default path is "./data"  
            print_stats (bool, optional): whether to print statistics of the dataset
        """
        super().__init__(name, path, print_stats,
                         dataset_names=dataset_names["DrugSyn"])

        if print_stats:
            self.print_stats()

        print('Done!', flush=True, file=sys.stderr)