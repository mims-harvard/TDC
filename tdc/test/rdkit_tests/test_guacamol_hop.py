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
                                             '../../..')))


class TestFunctions(unittest.TestCase):
    def setUp(self):
        print(os.getcwd())
        pass

    def test_Oracle(self):
        from tdc import Oracle
        oracle = Oracle(name = 'Hop')
        print(oracle(['CC(=O)OC1=CC=CC=C1C(=O)O',
               'C1=CC=C(C=C1)C=O']))

    def tearDown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), "data")):
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        if os.path.exists(os.path.join(os.getcwd(), "oracle")):
            shutil.rmtree(os.path.join(os.getcwd(), "oracle"))