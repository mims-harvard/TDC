# -*- coding: utf-8 -*-

import os
import sys

import unittest
import shutil
import pytest
import mygene

# temporary solution for relative imports in case TDC is not installed
# if TDC is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# TODO: add verification for the generation other than simple integration

from tdc.resource import cellxgene_census
from tdc.model_server.tokenizers.geneformer import GeneformerTokenizer

import requests


def get_ensembl_id(gene_symbols):
    mg = mygene.MyGeneInfo()
    return mg.querymany(gene_symbols, scopes='symbol', fields='ensembl.gene', species='human')


def get_target_from_chembl(chembl_id):
    # Query ChEMBL API for target information
    chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/{chembl_id}.json"
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

        adata = self.resource.get_anndata(
            var_value_filter = "feature_id in ['ENSG00000161798', 'ENSG00000188229']",
            obs_value_filter = "sex == 'female' and cell_type in ['microglial cell', 'neuron']",
            column_names = {"obs": ["assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease"]},
        )
        # adata.obs["ncounts"] = [2] * len(adata.obs)
        # raise Exception("obs", adata.obs.columns, "var", adata.var.columns)
        """
         Exception: ('obs', Index(['soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id',
       'cell_type', 'cell_type_ontology_term_id', 'development_stage',
       'development_stage_ontology_term_id', 'disease',
       'disease_ontology_term_id', 'donor_id', 'is_primary_data',
       'observation_joinid', 'self_reported_ethnicity',
       'self_reported_ethnicity_ontology_term_id', 'sex',
       'sex_ontology_term_id', 'suspension_type', 'tissue',
       'tissue_ontology_term_id', 'tissue_type', 'tissue_general',
       'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz',
       'raw_variance_nnz', 'n_measured_vars'],
      dtype='object'), 'var', Index(['soma_joinid', 'feature_id', 'feature_name', 'feature_length', 'nnz',
       'n_measured_obs'],
      dtype='object'))
        """
        print("initializing tokenizer")
        tokenizer = GeneformerTokenizer()
        print("testing tokenizer")
        x = tokenizer.tokenize_cell_vectors(adata, ensembl_id="feature_id", ncounts="n_measured_vars")
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
