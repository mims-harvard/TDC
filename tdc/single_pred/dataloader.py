# -*- coding: utf-8 -*-
"""Docstring to be finished.
"""
# Author: TDC Team
# License: MIT

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

    def __init__(self, name, path='./data', label_name=None, print_stats=False):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["ADME"])
        if print_stats:
            self.print_stats()
        print('Done!', flush = True, file = sys.stderr)

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

    def __init__(self, name, path='./data', label_name=None, print_stats=False):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["Toxicity"])
        if print_stats:
            self.print_stats()
        print('Done!', flush = True, file = sys.stderr)

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

    def __init__(self, name, path='./data', label_name=None, print_stats=False):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["Epitope"])
        self.entity1_name = 'Antigen'
        if print_stats:
            self.print_stats()
        print('Done!', flush = True, file = sys.stderr)

class Paratope(single_pred_dataset.DataLoader):
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

    def __init__(self, name, path='./data', label_name=None, print_stats=False):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["Paratope"])
        self.entity1_name = 'Antibody'
        if print_stats:
            self.print_stats()
        print('Done!', flush = True, file = sys.stderr)

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

    def __init__(self, name, path='./data', label_name=None, print_stats=False):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["HTS"])
        if print_stats:
            self.print_stats()
        print('Done!', flush = True, file = sys.stderr)

class Develop(single_pred_dataset.DataLoader):
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

    def __init__(self, name, path='./data', label_name=None, print_stats=False):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["Develop"])
        self.entity1_name = 'Antibody'
        if print_stats:
            self.print_stats()
        print('Done!', flush = True, file = sys.stderr)

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

    def __init__(self, name, path='./data', label_name=None, print_stats=False):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["QM"])
        if print_stats:
            self.print_stats()
        print('Done!', flush = True, file = sys.stderr)

class Yields(single_pred_dataset.DataLoader):
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

    def __init__(self, name, path='./data', label_name=None, print_stats=False):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["Yields"])
        self.entity1_name = 'Reaction'
        if print_stats:
            self.print_stats()
        print('Done!', flush = True, file = sys.stderr)

class CRISPROutcome(single_pred_dataset.DataLoader):
    """DNA repair outcomes following a CRISPR experiment.

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

    def __init__(self, name, path='./data', label_name=None, print_stats=False):
        super().__init__(name, path, label_name, print_stats,
                         dataset_names=dataset_names["CRISPROutcome"])
        self.entity1_name = 'GuideSeq'
        if print_stats:
            self.print_stats()
        print('Done!', flush = True, file = sys.stderr)
