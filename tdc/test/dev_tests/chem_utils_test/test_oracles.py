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


class TestOracle(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_Oracle(self):
        from tdc import Oracle

        oracle = Oracle(name="SA")
        x = oracle([
            "CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1",
            "CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1",
            "C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O",
        ])

        oracle = Oracle(name="Hop")
        x = oracle(["CC(=O)OC1=CC=CC=C1C(=O)O", "C1=CC=C(C=C1)C=O"])

    def test_distribution(self):
        from tdc import Evaluator

        evaluator = Evaluator(name="Diversity")
        x = evaluator([
            "CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1",
            "C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1",
            "CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1",
            "C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O",
        ])

    def tearDown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), "data")):
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        if os.path.exists(os.path.join(os.getcwd(), "oracle")):
            shutil.rmtree(os.path.join(os.getcwd(), "oracle"))
