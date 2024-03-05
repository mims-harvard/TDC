# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys

import unittest
import shutil

# temporary solution for relative imports in case TDC is not installed
# if TDC is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


class TestMolConvert(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_MolConvert(self):
        from tdc.chem_utils import MolConvert

        converter = MolConvert(src="SMILES", dst="Graph2D")
        converter([
            "Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC",
            "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C",
        ])

        from tdc.chem_utils import MolConvert

        MolConvert.eligible_format()

    #
    def tearDown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), "data")):
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
