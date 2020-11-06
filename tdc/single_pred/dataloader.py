# -*- coding: utf-8 -*-
"""Docstring to be finished.
"""
# Author:
# License: CC BY-NC-SA 4.0


import sys

import warnings

warnings.filterwarnings("ignore")

from . import single_pred_dataset
from ..utils import print_sys
from ..utils import train_val_test_split
from ..metadata import dataset_names


class ADME(single_pred_dataset.DataLoader):
    """Docstring to be finished.

    Parameters
    ----------
    name : str
        Description of the variable.

    path : str, optional (default="data")
        Description of the variable.

    label_name : str, optional (default=None)
        Description of the variable.

    print_stats : bool, optional (default=True)
        Description of the variable.
    """

    def __init__(self, name, path='./data', label_name=None, print_stats=True):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["ADME"])
        if print_stats:
            self.print_stats()

class Tox(single_pred_dataset.DataLoader):
    """Docstring to be finished.

    Parameters
    ----------
    name : str
        Description of the variable.

    path : str, optional (default="data")
        Description of the variable.

    label_name : str, optional (default=None)
        Description of the variable.

    print_stats : bool, optional (default=True)
        Description of the variable.
    """

    def __init__(self, name, path='./data', label_name=None, print_stats=True):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["Toxicity"])
        if print_stats:
            self.print_stats()

class Epitope(single_pred_dataset.DataLoader):
    """Docstring to be finished.

    Parameters
    ----------
    name : str
        Description of the variable.

    path : str, optional (default="data")
        Description of the variable.

    label_name : str, optional (default=None)
        Description of the variable.

    print_stats : bool, optional (default=True)
        Description of the variable.
    """

    def __init__(self, name, path='./data', label_name=None, print_stats=True):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["Epitope"])
        self.entity1_name = 'Antigen'
        if print_stats:
            self.print_stats()

class HTS(single_pred_dataset.DataLoader):
    """Docstring to be finished.

    Parameters
    ----------
    name : str
        Description of the variable.

    path : str, optional (default="data")
        Description of the variable.

    label_name : str, optional (default=None)
        Description of the variable.

    print_stats : bool, optional (default=True)
        Description of the variable.
    """

    def __init__(self, name, path='./data', label_name=None, print_stats=True):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["HTS"])
        if print_stats:
            self.print_stats()

### Not ready for reviews ###

class QM(single_pred_dataset.DataLoader):
    """Docstring to be finished.

    Parameters
    ----------
    name : str
        Description of the variable.

    path : str, optional (default="data")
        Description of the variable.

    label_name : str, optional (default=None)
        Description of the variable.

    print_stats : bool, optional (default=True)
        Description of the variable.
    """

    def __init__(self, name, path='./data', label_name=None, print_stats=True):
        try:
            entity1, y, entity1_idx = eval(
                name + '_process(name, path, label_name)')

            self.entity1 = entity1
            self.y = y
            self.entity1_idx = entity1_idx
            self.name = name
        except:
            raise AttributeError(
                "Please use the correct and available dataset name!")

        self.entity1_name = 'Drug'

        if print_stats:
            self.print_stats()

        print('Done!', flush=True, file=sys.stderr)

    def get_data(self, format='dict'):
        """Docstring to be finished.

        Parameters
        ----------
        format : str, optional (default="dict")
            Description of the variable.
        """
        if format == 'df':
            print_sys('the features are 2D distance map,'
                      'thus is not suitable for pandas, '
                      'switch to dictionary automatically...')
            format = 'dict'

        if format == 'dict':
            return {self.entity1_name + '_ID': self.entity1_idx,
                    self.entity1_name: self.entity1, 'Y': self.y}
        elif format == 'DeepPurpose':
            return self.entity1, self.y
        elif format == 'sklearn':
            pass
        else:
            raise AttributeError("Please use the correct format input")

    def get_split(self, method='random', seed='benchmark',
                  frac=[0.7, 0.1, 0.2]):
        """Docstring to be finished.

        Parameters
        ----------
        method : str, optional (default="random")
            Description of the variable.

        seed : str, optional (default="benchmark")
            Description of the variable.

        frac : list, optional (default=[0.7, 0.1, 0.2])
            Description of the variable.

        Returns
        -------
        data_split : dict
            Description of the variable.
        """

        # TODO: use seed=42 or 0?
        if seed == 'benchmark':
            seed = 1234

        df = self.get_data()

        if method == 'cold_' + self.entity1_name.lower():
            print_sys("cold drug is the same as random split for "
                      "drug property prediction...")
            method = 'random'

        if method == 'random':
            len_data = len(df['Drug_ID'])
            train, val, test = train_val_test_split(len_data, frac, seed)
            return {'X_train': df['Drug'][train], 'y_train': df['Y'][train],
                    'X_val': df['Drug'][val], 'y_val': df['Y'][val],
                    'X_test': df['Drug'][test], 'y_test': df['Y'][test]}
        elif method == 'scaffold':
            raise AttributeError(
                "Scaffold does not apply for QM dataset "
                "since the input features are 2D distance map")
        else:
            raise AttributeError("Please specify the correct splitting method")

    def print_stats(self):
        """Docstring to be finished.
        """
        print('There are ' + str(self.entity1.shape[0]) +
              ' unique ' + self.entity1_name.lower() + 's',
              flush=True, file=sys.stderr)
