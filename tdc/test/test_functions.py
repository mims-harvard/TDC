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

    def test_ADME(self):
        # Processing Helpers
        from tdc.single_pred import ADME
        data = ADME(name='Caco2_Wang')
        data.label_distribution()

    def test_binarize(self):
        from tdc.multi_pred import DTI
        data = DTI(name='DAVIS')
        data.binarize(threshold=30, order='descending')

    def test_convert_to_log(self):
        from tdc.multi_pred import DTI
        data = DTI(name='DAVIS')
        data.convert_to_log()

    def test_get_label_map(self):
        from tdc.multi_pred import DDI
        from tdc.utils import get_label_map
        data = DDI(name='DrugBank')
        split = data.get_split()
        get_label_map(name='DrugBank', task='DDI')

    def test_print_stats(self):
        from tdc.multi_pred import GDA
        data = GDA(name='DisGeNET')
        data.print_stats()

    def test_balanced(self):
        from tdc.single_pred import HTS
        data = HTS(name='SARSCoV2_3CLPro_Diamond')
        data.balanced(oversample=True, seed='benchmark')

    def test_to_graph(self):
        from tdc.multi_pred import DTI

        data = DTI(name='DAVIS')
        data.to_graph(threshold=30, format='edge_list', split=True,
                      frac=[0.7, 0.1, 0.2], seed='benchmark',
                      order='descending')
        # output: {'edge_list': array of shape (X, 2), 'neg_edges': array of shape (X, 2), 'split': {'train': df, 'valid': df, 'test': df}}

        data.to_graph(threshold=30, format='dgl', split=True,
                      frac=[0.7, 0.1, 0.2],
                      seed='benchmark', order='descending')
        # output: {'dgl_graph': the DGL graph object, 'index_to_entities': a dict map from ID in the data to node ID in the DGL object, 'split': {'train': df, 'valid': df, 'test': df}}

        data.to_graph(threshold=30, format='pyg', split=True,
                      frac=[0.7, 0.1, 0.2],
                      seed='benchmark', order='descending')
        # output: {'pyg_graph': the PyG graph object, 'index_to_entities': a dict map from ID in the data to node ID in the PyG object, 'split': {'train': df, 'valid': df, 'test': df}}

    def test_cid2smiles(self):
        from tdc.utils import cid2smiles
        smiles = cid2smiles(2248631)

    def test_uniprot2seq(self):
        from tdc.utils import uniprot2seq

        seq = uniprot2seq('P49122')

    def test_single_pred(self):
        # data split
        from tdc.single_pred import ADME

        data = ADME(name='Caco2_Wang')
        split = data.get_split(method='scaffold')

    def test_multi_pred(self):
        from tdc.multi_pred import DTI

        data = DTI(name='DAVIS')
        split = data.get_split(method='cold_split', column_name='Drug')

    def test_Oracle(self):
        # Molecule Generation Oracles

        from tdc import Oracle

        oracle = Oracle(name='GSK3B')
        smiles_lst = [
            'CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1', \
            'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']
        oracle(smiles_lst)

        oracle = Oracle(name='DRD2')
        smiles_lst = [
            'CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1', \
            'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']
        oracle(smiles_lst)

        oracle = Oracle(name='Hop')
        oracle(['CC(=O)OC1=CC=CC=C1C(=O)O',
                'C1=CC=C(C=C1)C=O'])

        oracle = Oracle(name='Valsartan_SMARTS')
        oracle(['CC(=O)OC1=CC=CC=C1C(=O)O',
                'C1=CC=C(C=C1)C=O'])

        oracle = Oracle(name='Rediscovery')
        oracle(['CC(=O)OC1=CC=CC=C1C(=O)O',
                'C1=CC=C(C=C1)C=O'])

        oracle = Oracle(name='SA')
        smiles_lst = [
            'CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1', \
            'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']
        oracle(smiles_lst)

        oracle = Oracle(name='Uniqueness')

        smiles_lst = [
            'CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1', \
            'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']

        oracle(smiles_lst)

        oracle = Oracle(name='Novelty')

        smiles_lst = [
            'CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1', \
            'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']

        oracle(smiles_lst, smiles_lst)

        oracle = Oracle(name='Diversity')

        smiles_lst = [
            'CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1', \
            'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']

        oracle(smiles_lst)

        oracle = Oracle(name='Scaffold Hop')
        oracle(['CC(=O)OC1=CC=CC=C1C(=O)O',
                'C1=CC=C(C=C1)C=O'])
