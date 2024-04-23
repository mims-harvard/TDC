# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")
import sys

from ..utils import print_sys
from .cellxgene import CellXGeneTemplate


class PerturbOutcome(CellXGeneTemplate):

    def __init__(self, name, path="./data", print_stats=False):
        super().__init__(name, path, print_stats)

    def get_mean_expression(self):
        raise ValueError("TODO")

    def get_DE_genes(self):
        raise ValueError("TODO")

    def get_dropout_genes(self):
        raise ValueError("TODO")
