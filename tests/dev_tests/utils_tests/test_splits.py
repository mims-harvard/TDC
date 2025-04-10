# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os

import unittest
import shutil

from tdc.multi_pred import DrugSyn
from tdc.multi_pred import DTI
## requires RDKit
from tdc.single_pred import ADME


class TestFunctions(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_random_split(self):

        data = ADME(name="Caco2_Wang")
        data.get_split(method="random")

    def test_scaffold_split(self):

        data = ADME(name="Caco2_Wang")
        data.get_split(method="scaffold")

    def test_cold_start_split(self):

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
        data = DrugSyn(name="DrugComb")
        data.get_split(method="combination")

    def test_time_split(self):
        data = DTI(name="BindingDB_Patent")
        data.get_split(method="time", time_column="Year")

    def test_tearDown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), "data")):
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
