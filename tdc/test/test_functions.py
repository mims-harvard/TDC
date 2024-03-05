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
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TestFunctions(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_Evaluator(self):
        from tdc import Evaluator

        evaluator = Evaluator(name="ROC-AUC")
        print(evaluator([0, 1], [0.5, 0.6]))

    def test_binarize(self):
        from tdc.single_pred import TestSinglePred

        data = TestSinglePred(name="Test_Single_Pred")
        data.binarize(threshold=-5, order="descending")

    def test_convert_to_log(self):
        from tdc.single_pred import TestSinglePred

        data = TestSinglePred(name="Test_Single_Pred")
        data.convert_to_log()

    def test_print_stats(self):
        from tdc.single_pred import TestSinglePred

        data = TestSinglePred(name="Test_Single_Pred")
        data.print_stats()

    def tearDown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), "data")):
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        if os.path.exists(os.path.join(os.getcwd(), "oracle")):
            shutil.rmtree(os.path.join(os.getcwd(), "oracle"))


if __name__ == "__main__":
    unittest.main()
