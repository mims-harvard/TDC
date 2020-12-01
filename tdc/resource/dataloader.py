import sys
import warnings
warnings.filterwarnings("ignore")

from . import dataset
from ..utils import print_sys
from ..utils import train_val_test_split
from ..metadata import dataset_names

class CompoundLibrary(dataset.DataLoader):
    def __init__(self, name, path='./data', print_stats=False):
        super().__init__(name, path, print_stats,
                         dataset_names=dataset_names["CompoundLibrary"])

        if print_stats:
            self.print_stats()

        print('Done!', flush=True, file=sys.stderr)

class BioKG(dataset.DataLoader):
    def __init__(self, name, path='./data', print_stats=False):
        super().__init__(name, path, print_stats,
                         dataset_names=dataset_names["BioKG"])

        if print_stats:
            self.print_stats()

        print('Done!', flush=True, file=sys.stderr)