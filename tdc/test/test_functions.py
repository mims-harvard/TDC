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


class TestLabelUtils(unittest.TestCase):
    """Unit conversions in tdc.utils.label (#392)."""

    def test_p_to_nm(self):
        import numpy as np

        from tdc.utils.label import convert_y_unit

        # p=9 corresponds to ~1 nM; p=6 to ~1000 nM
        out = convert_y_unit(np.array([9.0, 6.0]), "p", "nM")
        self.assertTrue(np.allclose(out, [1.0, 1000.0], atol=0.2))

    def test_nm_to_p_roundtrip(self):
        import numpy as np

        from tdc.utils.label import convert_y_unit

        y = np.array([1.0, 10.0, 1000.0])
        p = convert_y_unit(y, "nM", "p")
        back = convert_y_unit(p, "p", "nM")
        self.assertTrue(np.allclose(back, y, rtol=1e-3))

    def test_convert_back_log_helper(self):
        import numpy as np

        from tdc.utils.label import convert_back_log, convert_to_log

        y = np.array([5.0, 50.0, 500.0])
        self.assertTrue(
            np.allclose(convert_back_log(convert_to_log(y)), y, rtol=1e-3))
