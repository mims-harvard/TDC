"""Summary
"""
import warnings
warnings.filterwarnings("ignore")
import sys

from ..utils import print_sys
from . import bi_pred_dataset, multi_pred_dataset
from ..metadata import dataset_names


class DDI(bi_pred_dataset.DataLoader):

    """Summary
    
    Attributes:
        entity1_name (str): Description
        entity2_name (str): Description
        two_types (bool): Description
    """
    
    def __init__(self, name, path='./data', label_name=None,
                 print_stats=False):
        """Summary
        
        Args:
            name (TYPE): Description
            path (str, optional): Description
            label_name (None, optional): Description
            print_stats (bool, optional): Description
        """
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["DDI"])
        self.entity1_name = 'Drug1'
        self.entity2_name = 'Drug2'
        self.two_types = False

        if print_stats:
            self.print_stats()

        print('Done!', flush=True, file=sys.stderr)

    def print_stats(self):
        """Summary
        """
        print_sys('--- Dataset Statistics ---')
        print('There are ' + str(len(np.unique(
            self.entity1.tolist() + self.entity2.tolist()))) + ' unique drugs.',
              flush=True, file=sys.stderr)
        print('There are ' + str(len(self.y)) + ' drug-drug pairs.',
              flush=True, file=sys.stderr)
        print_sys('--------------------------')
