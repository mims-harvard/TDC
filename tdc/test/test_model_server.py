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
        print("initializing tokenizer")
        tokenizer = GeneformerTokenizer()
        print("testing tokenizer")
        x = tokenizer.tokenize_cell_vectors(adata, ensembl_id="feature_id", ncounts="n_measured_vars")
        assert x[0]

        # test Geneformer can serve the request
        cells, metadata = x
        assert cells, "FAILURE: cells false-like. Value is = {}".format(cells)
        assert len(cells) > 0, "FAILURE: length of cells <= 0 {}".format(cells)
        from tdc import tdc_hf_interface
        # import torch
        geneformer = tdc_hf_interface("Geneformer")
        model = geneformer.load()
        tokenized_data = tokenizer.create_dataset(cells, metadata)
        out = model(tokenized_data)
        # input_tensor = torch.tensor(cells)
        # input_tensor_squeezed = torch.squeeze(input_tensor)
        # x = input_tensor_squeezed.shape[0]
        # y = input_tensor_squeezed.shape[1]
        # out = None  # try-except block
        # try:
        #     input_tensor_squeezed = input_tensor_squeezed.reshape(x, y)
        #     out = model(input_tensor_squeezed)
        # except Exception as e:
        #     raise Exception("tensor shape is", input_tensor.shape, "exception was: {}".format(e), "input_tensor_squeezed is\n", input_tensor, "\n\ninput_tensor normal is: {}".format(input_tensor))
        assert out, "FAILURE: Geneformer output is false-like. Value = {}".format(out)
        assert out.shape[0] == input_tensor.shape[0], "FAILURE: Geneformer output and input tensor input don't have the same length. {} vs {}".format(out.shape[0], input_tensor.shape[0])
        assert out.shape[0] == len(cells), "FAILURE: Geneformer output and tokenized cells don't have the same length. {} vs {}".format(out.shape[0], len(cells))

    def tearDown(self):
        try:
            print(os.getcwd())
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        except:
            pass
