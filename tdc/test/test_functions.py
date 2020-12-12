# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys

import unittest
import shutil

# temporary solution for relative imports in case TDC is not installed
# if TDC is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))

class TestFunctions(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_Evaluator(self):
        from tdc import Evaluator
        evaluator = Evaluator(name='ROC-AUC')
        print(evaluator([0, 1], [0.5, 0.6]))

    # TODO: commented out because plotting func.
    # def test_ADME(self):
    #     # Processing Helpers
    #     from tdc.single_pred import ADME
    #     data = ADME(name='Caco2_Wang')
    #     data.label_distribution()

    def test_binarize(self):
        from tdc.single_pred import ADME
        data = ADME(name = 'Caco2_Wang')
        data.binarize(threshold=-5, order='descending')

    def test_convert_to_log(self):
        from tdc.single_pred import ADME
        data = ADME(name = 'Caco2_Wang')
        data.convert_to_log()

    #def test_get_label_map(self):
    #    from tdc.multi_pred import DDI
    #    from tdc.utils import get_label_map
    #    data = DDI(name='DrugBank')
    #    split = data.get_split()
    #    get_label_map(name='DrugBank', task='DDI')

    def test_print_stats(self):
        from tdc.single_pred import ADME
        data = ADME(name = 'Caco2_Wang')
        data.print_stats()

    def test_balanced(self):
        from tdc.single_pred import HTS
        data = HTS(name='SARSCoV2_3CLPro_Diamond')
        data.balanced(oversample=True, seed='benchmark')

    # TODO: fix dependency later
    # def test_to_graph(self):
    #     from tdc.multi_pred import DTI
    #
    #     data = DTI(name='DAVIS')
    #     data.to_graph(threshold=30, format='edge_list', split=True,
    #                   frac=[0.7, 0.1, 0.2], seed='benchmark',
    #                   order='descending')
    #     # output: {'edge_list': array of shape (X, 2), 'neg_edges': array of shape (X, 2), 'split': {'train': df, 'valid': df, 'test': df}}
    #
    #     data.to_graph(threshold=30, format='dgl', split=True,
    #                   frac=[0.7, 0.1, 0.2],
    #                   seed='benchmark', order='descending')
    #     # output: {'dgl_graph': the DGL graph object, 'index_to_entities': a dict map from ID in the data to node ID in the DGL object, 'split': {'train': df, 'valid': df, 'test': df}}
    #
    #     data.to_graph(threshold=30, format='pyg', split=True,
    #                   frac=[0.7, 0.1, 0.2],
    #                   seed='benchmark', order='descending')
    #     # output: {'pyg_graph': the PyG graph object, 'index_to_entities': a dict map from ID in the data to node ID in the PyG object, 'split': {'train': df, 'valid': df, 'test': df}}

    # def test_cid2smiles(self):
    #   from tdc.utils import cid2smiles
    #    smiles = cid2smiles(2248631)

    #def test_uniprot2seq(self):
    #    from tdc.utils import uniprot2seq

    #    seq = uniprot2seq('P49122')

    def tearDown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), "data")):
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        if os.path.exists(os.path.join(os.getcwd(), "oracle")):
            shutil.rmtree(os.path.join(os.getcwd(), "oracle"))