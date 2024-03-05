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


class TestMolFilter(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_MolConvert(self):
        from tdc.chem_utils import MolFilter

        filters = MolFilter(filters=["PAINS"], HBD=[0, 6])
        filters(["CCSc1ccccc1C(=O)Nc1onc2c1CCC2"])

    #
    def tearDown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), "data")):
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
