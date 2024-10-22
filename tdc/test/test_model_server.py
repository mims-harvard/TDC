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

import requests


def get_target_from_chembl(chembl_id):
    # Query ChEMBL API for target information
    chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json"
    response = requests.get(chembl_url)

    if response.status_code == 200:
        data = response.json()
        # Extract UniProt ID from the ChEMBL target info
        for component in data.get('target_components', []):
            for xref in component.get('target_component_xrefs', []):
                if xref['xref_src_db'] == 'UniProt':
                    return xref['xref_id']
    else:
        raise ValueError(f"ChEMBL ID {chembl_id} not found or invalid.")
    return None


def get_ensembl_from_uniprot(uniprot_id):
    # Query UniProt API to get Ensembl ID from UniProt ID
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(uniprot_url)

    if response.status_code == 200:
        data = response.json()
        # Extract Ensembl Gene ID from the cross-references
        for xref in data.get('dbReferences', []):
            if xref['type'] == 'Ensembl':
                return xref['id']
    else:
        raise ValueError(f"UniProt ID {uniprot_id} not found or invalid.")
    return None


def get_ensembl_id_from_chembl_id(chembl_id):
    try:
        # Step 1: Get UniProt ID from ChEMBL
        uniprot_id = get_target_from_chembl(chembl_id)
        if not uniprot_id:
            return f"No UniProt ID found for ChEMBL ID {chembl_id}"

        # Step 2: Get Ensembl ID from UniProt
        ensembl_id = get_ensembl_from_uniprot(uniprot_id)
        if not ensembl_id:
            return f"No Ensembl ID found for UniProt ID {uniprot_id}"

        return f"Ensembl ID for ChEMBL ID {chembl_id}: {ensembl_id}"
    except Exception as e:
        return str(e)


class TestModelServer(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        self.resource = cellxgene_census.CensusResource()

    def testGeneformerTokenizer(self):
        import anndata
        from tdc.multi_pred.perturboutcome import PerturbOutcome
        test_loader = PerturbOutcome(
            name="scperturb_drug_AissaBenevolenskaya2021")
        adata = test_loader.adata
        print("swapping obs and var because scperturb violated convention...")
        adata_flipped = anndata.AnnData(adata.X.T)
        adata_flipped.obs = adata.var
        adata_flipped.var = adata.obs
        adata = adata_flipped
        print("swap complete")
        print("adding ensembl ids...")
        adata.var["ensembl_id"] = adata.var["chembl-ID"].apply(
            get_ensembl_id_from_chembl_id)
        print("added ensembl_id column")

        print(type(adata.var))
        print(adata.var.columns)
        print(type(adata.obs))
        print(adata.obs.columns)
        print("initializing tokenizer")
        tokenizer = GeneformerTokenizer()
        print("testing tokenizer")
        x = tokenizer.tokenize_cell_vectors(adata)
        assert x[0]

        # test Geneformer can serve the request
        cells = x[0]
        print("cells is", len(cells), cells)
        assert cells[0]
        assert len(cells[0]) > 0
        from tdc import tdc_hf_interface
        import torch
        geneformer = tdc_hf_interface("Geneformer")
        model = geneformer.load()
        out = model(torch.tensor(cells))
        assert out
        assert out[0]
        assert len(out[0]) > 0

    def tearDown(self):
        try:
            print(os.getcwd())
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        except:
            pass
