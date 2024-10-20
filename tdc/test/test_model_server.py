# -*- coding: utf-8 -*-

import os
import sys

import unittest
import shutil
import pytest

# temporary solution for relative imports in case TDC is not installed
# if TDC is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# TODO: add verification for the generation other than simple integration

from tdc.resource import cellxgene_census
from tdc.model_server.tokenizers.geneformer import GeneformerTokenizer 

class TestModelServer(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        self.resource = cellxgene_census.CensusResource()

    def testGeneformerTokenizer(self):
        # genes = ['ENSG00000161798', 'ENSG00000188229']
        # cell_types = ['mucus secreting cell', 'neuroendocrine cell']
        # obs_cols = ["dataset_id", "assay", "suspension_type", "sex", "tissue_general", "tissue", "cell_type", "ncounts"]
        # adata = self.resource.gget_czi_cellxgene(
        #     ensembl=True,
        #     gene=genes,
        #     cell_type=cell_types,
        #     column_names=obs_cols,
        # )
        # TODO: scperturb is using chembl, NOT ensembl. geneformer assumes ensembl. can fix by going back to cellxgene and not normalizing
        from tdc.multi_pred.perturboutcome import PerturbOutcome
        test_loader = PerturbOutcome(
            name="scperturb_drug_AissaBenevolenskaya2021")
        adata = test_loader.adata
        print(type(adata.var))
        print(adata.var.columns)
        print(type(adata.obs))
        print(adata.obs.columns)
        print("initializing tokenizer")
        tokenizer = GeneformerTokenizer()
        print("testing tokenizer")
        x = tokenizer.tokenize_cell_vectors(adata)
        assert x 

    def tearDown(self):
        try:
            print(os.getcwd())
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        except:
            pass

    