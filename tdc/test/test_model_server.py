# -*- coding: utf-8 -*-

import os
import sys

import unittest
import shutil
import pytest
import mygene
import numpy as np

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
    return mg.querymany(gene_symbols,
                        scopes='symbol',
                        fields='ensembl.gene',
                        species='human')


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
            var_value_filter=
            "feature_id in ['ENSG00000161798', 'ENSG00000188229']",
            obs_value_filter=
            "sex == 'female' and cell_type in ['microglial cell', 'neuron']",
            column_names={
                "obs": [
                    "assay", "cell_type", "tissue", "tissue_general",
                    "suspension_type", "disease"
                ]
            },
        )
        tokenizer = GeneformerTokenizer()
        x = tokenizer.tokenize_cell_vectors(adata,
                                            ensembl_id="feature_id",
                                            ncounts="n_measured_vars")
        assert x[0]

        # test Geneformer can serve the request
        cells, _ = x
        assert cells, "FAILURE: cells false-like. Value is = {}".format(cells)
        assert len(cells) > 0, "FAILURE: length of cells <= 0 {}".format(cells)
        from tdc import tdc_hf_interface
        import torch
        geneformer = tdc_hf_interface("Geneformer")
        model = geneformer.load()

        # using very few genes for these test cases so expecting empties... let's pad...
        for idx in range(len(cells)):
            x = cells[idx]
            for j in range(len(x)):
                v = x[j]
                if len(v) < 2:
                    out = None
                    for _ in range(2 - len(v)):
                        if out is None:
                            out = np.append(v, 0)  # pad with 0
                        else:
                            out = np.append(out, 0)
                    cells[idx][j] = out
            if len(cells[idx]) < 512:  # batch size
                array = cells[idx]
                # Calculate how many rows need to be added
                n_rows_to_add = 512 - len(array)

                # Create a padding array with [0, 0] for the remaining rows
                padding = np.tile([0, 0], (n_rows_to_add, 1))

                # Concatenate the original array with the padding array
                cells[idx] = np.vstack((array, padding))

        input_tensor = torch.tensor(cells)
        out = []
        try:
            ctr = 0  # stop after some passes to avoid failure
            for batch in input_tensor:
                # build an attention mask
                attention_mask = torch.tensor(
                    [[x[0] != 0, x[1] != 0] for x in batch])
                out.append(model(batch, attention_mask=attention_mask))
                if ctr == 2:
                    break
                ctr += 1
        except Exception as e:
            raise Exception(e)

        assert out, "FAILURE: Geneformer output is false-like. Value = {}".format(
            out)
        assert len(
            out
        ) == 3, "length not matching ctr+1: {} vs {}. output was \n {}".format(
            len(out), ctr + 1, out)

    def tearDown(self):
        try:
            print(os.getcwd())
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        except:
            pass
