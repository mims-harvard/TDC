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


class TestDataloader(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_single_pred(self):
        from tdc.single_pred import TestSinglePred

        data = TestSinglePred(name="Test_Single_Pred")
        split = data.get_split()

    def test_multi_pred(self):
        from tdc.multi_pred import TestMultiPred

        data = TestMultiPred(name="Test_Multi_Pred")
        split = data.get_split()

    def test_pentelute(self):
        # TODO: factor out into specialized test suites for individual datasets
        # this test serves as an integration test of the data processing, data configs, and existing tdc pipeline. leave here for now.
        from tdc.multi_pred import ProteinPeptide
        data = ProteinPeptide(name="pentelute_mdm2_ace2_12ca5")
        assert "protein_or_rna_sequence" in data.get_data(
        ).columns  # pentelute protein<>peptide dataset uses a data config inserting this column
        data.get_split()

    @unittest.skip(
        "test is taking up too much memory"
    )  #FIXME: should probably create much smaller version and use that for the test. This test does pass locally, please rerun if changing anndata code.
    def test_h5ad_dataloader(self):
        from tdc.multi_pred.cellxgene import SingleCellPrediction
        from pandas import DataFrame
        test_loader = SingleCellPrediction(
            name="scperturb_drug_AissaBenevolenskaya2021")
        testdf = test_loader.get_data()
        assert isinstance(testdf, DataFrame)
        test_loader.get_split()

    def test_generation(self):
        from tdc.generation import MolGen

        data = MolGen(name="ZINC")
        split = data.get_split()

    def tearDown(self):
        print(os.getcwd())
        shutil.rmtree(os.path.join(os.getcwd(), "data"))


if __name__ == "__main__":
    unittest.main()
