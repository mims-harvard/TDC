# -*- coding: utf-8 -*-

import os
import sys

import unittest
import shutil
import numpy as np

# temporary solution for relative imports in case TDC is not installed
# if TDC is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# TODO: add verification for the generation other than simple integration

from tdc.resource import cellxgene_census
from tdc.model_server.tokenizers.geneformer import GeneformerTokenizer


def quant_layers(model):
    layer_nums = []
    for name, parameter in model.named_parameters():
        if "layer" in name:
            layer_nums += [int(name.split("layer.")[1].split(".")[0])]
    return int(max(layer_nums)) + 1


class TestModelServer(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        self.resource = cellxgene_census.CensusResource()

    def testscGPT(self):
        from tdc.multi_pred.anndata_dataset import DataLoader
        from tdc import tdc_hf_interface
        from tdc.model_server.tokenizers.scgpt import scGPTTokenizer
        import torch
        adata = DataLoader("cellxgene_sample_small",
                           "./data",
                           dataset_names=["cellxgene_sample_small"],
                           no_convert=True).adata
        scgpt = tdc_hf_interface("scGPT")
        model = scgpt.load()  # this line can cause segmentation fault
        tokenizer = scGPTTokenizer()
        gene_ids = adata.var["feature_name"].to_numpy(
        )  # Convert to numpy array
        tokenized_data = tokenizer.tokenize_cell_vectors(
            adata.X.toarray(), gene_ids)
        mask = torch.tensor([x != 0 for x in tokenized_data[0][1]],
                            dtype=torch.bool)
        assert sum(mask) != 0, "FAILURE: mask is empty"
        first_embed = model(tokenized_data[0][0],
                            tokenized_data[0][1],
                            attention_mask=mask)
        print(f"scgpt ran successfully. here is an output {first_embed}")

    def testGeneformerPerturb(self):
        from tdc.multi_pred.perturboutcome import PerturbOutcome
        dataset = "scperturb_drug_AissaBenevolenskaya2021"
        data = PerturbOutcome(dataset)
        adata = data.adata
        tokenizer = GeneformerTokenizer(max_input_size=3)
        adata.var["feature_id"] = adata.var.index.map(
            lambda x: tokenizer.gene_name_id_dict.get(x, 0))
        x = tokenizer.tokenize_cell_vectors(adata,
                                            ensembl_id="feature_id",
                                            ncounts="ncounts")
        cells, _ = x
        assert cells, "FAILURE: cells false-like. Value is = {}".format(cells)
        assert len(cells) > 0, "FAILURE: length of cells <= 0 {}".format(cells)
        from tdc import tdc_hf_interface
        import torch
        geneformer = tdc_hf_interface("Geneformer")
        model = geneformer.load()
        mdim = max(len(cell) for b in cells for cell in b)
        batch = cells[0]
        for idx, cell in enumerate(batch):
            if len(cell) < mdim:
                for _ in range(mdim - len(cell)):
                    cell = np.append(cell, 0)
                batch[idx] = cell
        input_tensor = torch.tensor(batch)
        assert input_tensor.shape[0] == 512, "unexpected batch size"
        assert input_tensor.shape[1] == mdim, f"unexpected gene length {mdim}"
        attention_mask = torch.tensor([[t != 0 for t in cell] for cell in batch
                                      ])
        assert input_tensor.shape[0] == attention_mask.shape[0]
        assert input_tensor.shape[1] == attention_mask.shape[1]
        try:
            outputs = model(input_tensor,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        except Exception as e:
            raise Exception(
                f"sizes: {input_tensor.shape[0]}, {input_tensor.shape[1]}\n {e}"
            )
        num_out_in_batch = len(outputs.hidden_states[-1])
        input_batch_size = input_tensor.shape[0]
        num_gene_out_in_batch = len(outputs.hidden_states[-1][0])
        assert num_out_in_batch == input_batch_size, f"FAILURE: length doesn't match batch size {num_out_in_batch} vs {input_batch_size}"
        assert num_gene_out_in_batch == mdim, f"FAILURE: out length {num_gene_out_in_batch} doesn't match gene length {mdim}"

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
                outputs = model(batch,
                                attention_mask=attention_mask,
                                output_hidden_states=True)
                layer_to_quant = quant_layers(model) + (
                    -1
                )  # TODO note this can be parametrized to either 0 (extract last embedding layer) or -1 (second-to-last which is more generalized)
                embs_i = outputs.hidden_states[layer_to_quant]
                # there are "cls", "cell", and "gene" embeddings. we will only capture "gene", which is cell type specific. for "cell", you'd average out across unmasked gene embeddings per cell
                embs = embs_i
                out.append(embs)
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
        print(
            "Geneformer ran sucessfully. Find batch embedding example here:\n {}"
            .format(out[0]))

    def tearDown(self):
        try:
            print(os.getcwd())
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        except:
            pass
