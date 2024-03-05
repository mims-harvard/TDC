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
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


class TestFunctions(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_random_split(self):
        from tdc.single_pred import ADME

        data = ADME(name="Caco2_Wang")
        split = data.get_split(method="random")

    def test_scaffold_split(self):
        ## requires RDKit
        from tdc.single_pred import ADME

        data = ADME(name="Caco2_Wang")
        split = data.get_split(method="scaffold")

    def test_cold_start_split(self):
        from tdc.multi_pred import DTI

        data = DTI(name="DAVIS")
        split = data.get_split(method="cold_split", column_name="Drug")

        self.assertEqual(
            0,
            len(
                set(split["train"]["Drug"]).intersection(
                    set(split["test"]["Drug"]))))
        self.assertEqual(
            0,
            len(
                set(split["valid"]["Drug"]).intersection(
                    set(split["test"]["Drug"]))))
        self.assertEqual(
            0,
            len(
                set(split["train"]["Drug"]).intersection(
                    set(split["valid"]["Drug"]))),
        )

        multi_split = data.get_split(method="cold_split",
                                     column_name=["Drug_ID", "Target_ID"])
        for entity in ["Drug_ID", "Target_ID"]:
            train_entity = set(multi_split["train"][entity])
            valid_entity = set(multi_split["valid"][entity])
            test_entity = set(multi_split["test"][entity])
            self.assertEqual(0, len(train_entity.intersection(valid_entity)))
            self.assertEqual(0, len(train_entity.intersection(test_entity)))
            self.assertEqual(0, len(valid_entity.intersection(test_entity)))

    def test_combination_split(self):
        from tdc.multi_pred import DrugSyn

        data = DrugSyn(name="DrugComb")
        split = data.get_split(method="combination")

    def test_time_split(self):
        from tdc.multi_pred import DTI

        data = DTI(name="BindingDB_Patent")
        split = data.get_split(method="time", time_column="Year")

    def test_tearDown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), "data")):
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
