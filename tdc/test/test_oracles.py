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
# TODO: add verification for the generation other than simple integration


class TestOracle(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_gsk3b(self):
        from tdc import Oracle

        oracle = Oracle(name='GSK3B')

        x = oracle('CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1')
        assert abs(x - 0.03) < 0.0001

    def test_jnk3(self):
        from tdc import Oracle

        oracle = Oracle(name='JNK3')

        x = oracle('C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O')
        assert abs(x - 0.01) < 0.0001

    def test_list_single(self):
        from tdc import Oracle

        oracle = Oracle(name='GSK3B')

        x = oracle(['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1'])
        assert abs(x[0] - 0.03) < 0.0001

    def test_list_multi(self):
        from tdc import Oracle

        oracle = Oracle(name='JNK3')

        x = oracle(['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
    'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
    'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O'])
        assert abs(x[0] - 0.01) < 0.0001
        assert abs(x[1] - 0.0) < 0.0001
        assert abs(x[2] - 0.01) < 0.0001

    def test_oracle_update(self):
        from tdc import Oracle
        oracle = Oracle(name='DRD2')
        y = oracle(['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
                'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O'])
        assert abs(y[0] - 0.0015465365340340924) <= 1e-10
        assert abs(y[1] - 0.0023541754878916416) <= 1e-10
        assert abs(y[2] - 0.004715407010872501) <= 1e-10

    # def tearDown(self):
    #     print(os.getcwd())
    #     shutil.rmtree(os.path.join(os.getcwd(), "data"))


if __name__ == "__main__":
    unittest.main()
