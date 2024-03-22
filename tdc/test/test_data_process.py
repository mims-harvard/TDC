# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys

import unittest
import shutil

import pandas as pd

# temporary solution for relative imports in case TDC is not installed
# if TDC is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tdc.feature_generators.data_feature_generator import DataFeatureGenerator
from tdc.feature_generators.protein_feature_generator import ProteinFeatureGenerator


class TestDataFeatureGenerator(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def testAutofill(self):
        test_entries = [[0, "x", 8], [1, 'y', 4], [None, "x", 9],
                        [None, "y", 8], [2, "z", 12]]
        col_names = ["autofill", "index", "value"]
        df = pd.DataFrame(test_entries, columns=col_names)
        df2 = DataFeatureGenerator.autofill_identifier(df, "autofill", "index")
        self.assertEqual(df["autofill"].tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(df2["autofill"].tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(df2["index"].tolist(), ["x", "y", "x", "y", "z"])
        self.assertEqual(df2["value"].tolist(), [8, 4, 9, 8, 12])
        self.assertEqual(df2.shape[0], 5)
        self.assertEqual(df2.shape[1], 3)

    def testCreateRange(self):
        test_entries = [["7.7±4.5", 0], ["10±2.3", 1], ["Putative binder", 5]]
        col_names = ["num", "some_value"]
        keys = ["Putative binder"]
        subs = [0]
        df = pd.DataFrame(test_entries, columns=col_names)
        df2 = DataFeatureGenerator.create_range(df, "num", keys, subs)
        assert "expected" in df.columns
        assert "expected" in df2.columns
        assert "lower" in df2.columns
        assert "upper" in df2.columns
        self.assertEqual(df2["expected"].tolist(), [7.7, 10, 0])
        self.assertEqual(df2["lower"].tolist(), [3.2, 7.7, 0])
        self.assertEqual(df2["upper"].tolist(), [12.2, 12.3, 0])
        self.assertEqual(df2["num"].tolist(),
                         ["7.7±4.5", "10±2.3", "Putative binder"])
        self.assertEqual(df2["some_value"].tolist(), [0, 1, 5])
        self.assertEqual(df2.shape[0], 3)
        self.assertEqual(df2.shape[1], 5)

    def testProcessData(self):
        test_entries1 = [
            ["7.7±4.5", 0],
            ["10±2.3", 1],
            ["Putative binder", 5],
            ["Putative binder", 5],
            ["Putative binder", 5],
        ]
        test_entries2 = [[0, "x", 8], [1, 'y', 4], [None, "x", 9],
                         [None, "y", 8], [2, "z", 12]]
        test_entries = [x + y for x, y in zip(test_entries1, test_entries2)]
        col_names = [
            "num",
            "some_value",
            "autofill",
            "index",
            "value",
        ]
        assert len(col_names) == len(test_entries[0]),\
            ("Number of columns in test_entries does not match number of columns in col_names", col_names, test_entries[0])
        functions = [
            "create_range",
            "autofill_identifier",
        ]
        args = [
            {
                "column": "num",
                "keys": ["Putative binder"],
                "subs": [0],
            },
            {
                "autofill_column": "autofill",
                "key_column": "index"
            },
        ]
        df = pd.DataFrame(test_entries, columns=col_names)
        df2 = DataFeatureGenerator.process_data(df, functions, args)
        assert "expected" in df.columns
        assert "expected" in df2.columns
        assert "lower" in df2.columns
        assert "upper" in df2.columns
        self.assertEqual(df2["expected"].tolist(), [7.7, 10, 0, 0, 0])
        self.assertEqual(df2["lower"].tolist(), [3.2, 7.7, 0, 0, 0])
        self.assertEqual(df2["upper"].tolist(), [12.2, 12.3, 0, 0, 0])
        self.assertEqual(df2["num"].tolist(), [
            "7.7±4.5", "10±2.3", "Putative binder", "Putative binder",
            "Putative binder"
        ])
        self.assertEqual(df2["some_value"].tolist(), [0, 1, 5, 5, 5])
        self.assertEqual(df2.shape[0], 5)
        self.assertEqual(df2.shape[1], 8)
        self.assertEqual(df["autofill"].tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(df2["autofill"].tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(df2["index"].tolist(), ["x", "y", "x", "y", "z"])
        self.assertEqual(df2["value"].tolist(), [8, 4, 9, 8, 12])

    def tearDown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), "data")):
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        if os.path.exists(os.path.join(os.getcwd(), "oracle")):
            shutil.rmtree(os.path.join(os.getcwd(), "oracle"))


class TestProteinDataUtil(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def testInsertProteinSequence(self):
        test_entries = [
            ["BRCA1"],
            ["MDM2"],
            ["ACE2"],
            ["12CA5"],
        ]
        col_names = ["Gene name"]
        df = pd.DataFrame(test_entries, columns=col_names)
        df2 = ProteinFeatureGenerator.insert_protein_sequence(df, "Gene name")
        self.assertEqual(df2.shape[0], 4)
        self.assertEqual(df2.shape[1], 3)
        self.assertEqual(df2["Gene name"].tolist(),
                         ["BRCA1", "MDM2", "ACE2", "12CA5"])
        self.assertEqual(len(df2["protein_or_rna_sequence"]), 4)
        self.assertEqual(df2["protein_or_rna_sequence"].tolist(),
                         df["protein_or_rna_sequence"].tolist())
        self.assertEqual(len(df2["protein_or_rna_sequence"].unique()), 4)

    def tearDown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), "data")):
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        if os.path.exists(os.path.join(os.getcwd(), "oracle")):
            shutil.rmtree(os.path.join(os.getcwd(), "oracle"))


if __name__ == "__main__":
    unittest.main()
