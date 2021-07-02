# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys

import unittest
import shutil

# temporary solution for relative imports in case TDC is not installed
# if TDC is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
#TODO: add verification for the generation other than simple integration

class TestDataloader(unittest.TestCase):
    def setUp(self):
        print(os.getcwd())
        pass

    def test_single_pred(self):
        from tdc.single_pred import TestSinglePred
        data = TestSinglePred(name='Test_Single_Pred')
        split = data.get_split()

    def test_multi_pred(self):
        from tdc.multi_pred import TestMultiPred
        data = TestMultiPred(name='Test_Multi_Pred')
        split = data.get_split()

    def tearDown(self):
        print(os.getcwd())
        shutil.rmtree(os.path.join(os.getcwd(), "data"))
