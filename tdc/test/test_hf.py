# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

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


class TestHF(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    @pytest.mark.skip(
        reason="This test is skipped due to deeppurpose installation dependency"
    )
    @unittest.skip(reason="DeepPurpose")
    def test_hf_load_predict(self):
        from tdc.single_pred import Tox
        data = Tox(name='herg_karim')

        from tdc import tdc_hf_interface
        tdc_hf = tdc_hf_interface("hERG_Karim-CNN")
        # load deeppurpose model from this repo
        dp_model = tdc_hf.load_deeppurpose('./data')
        tdc_hf.predict_deeppurpose(dp_model, ['CC(=O)NC1=CC=C(O)C=C1'])

    def test_hf_transformer(self):
        from tdc import tdc_hf_interface
        # from transformers import Pipeline
        from transformers import BertForMaskedLM as BertModel
        geneformer = tdc_hf_interface("Geneformer")
        model = geneformer.load()
        # assert isinstance(pipeline, Pipeline)
        assert isinstance(model, BertModel), type(model)

    # def test_hf_load_new_pytorch_standard(self):
    #     from tdc import tdc_hf_interface
    #     # from tdc.resource.dataloader import DataLoader
    #     # data = DataLoader(name="pinnacle_dti")
    #     tdc_hf = tdc_hf_interface("mli-PINNACLE")
    #     dp_model = tdc_hf.load()
    #     assert dp_model is not None

    def tearDown(self):
        try:
            print(os.getcwd())
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        except:
            pass


if __name__ == "__main__":
    unittest.main()
