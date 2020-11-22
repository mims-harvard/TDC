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
#TODO: add verification for the generation other than simple integration

class TestDataloader(unittest.TestCase):
    def setUp(self):
        print(os.getcwd())
        pass

    # ADME, HTS
    def test_ADME(self):
        from tdc.single_pred import ADME
        data = ADME(name='Caco2_Wang')
        split = data.get_split()

    # Tox
    def test_Tox(self):
        from tdc.utils import retrieve_label_name_list
        label_list = retrieve_label_name_list('Tox21')
        from tdc.single_pred import Tox
        data = Tox(name='Tox21', label_name=label_list[0])
        split = data.get_split()

    # QM
    def test_QM(self):
        from tdc.utils import retrieve_label_name_list
        label_list = retrieve_label_name_list('QM7b')
        from tdc.single_pred import QM
        data = QM(name='QM7b', label_name=label_list[0])
        split = data.get_split()

    # Yields
    def test_Yields(self):
        from tdc.single_pred import Yields
        data = Yields(name='Buchwald-Hartwig')
        split = data.get_split()

    # Paratope Epitope
    def test_Paratope(self):
        from tdc.single_pred import Paratope
        data = Paratope(name='SAbDab_Liberis')
        split = data.get_split()

    # Develop
    def test_Develop(self):
        from tdc.utils import retrieve_label_name_list
        label_list = retrieve_label_name_list('TAP')

        from tdc.single_pred import Develop
        data = Develop(name='TAP', label_name=label_list[0])
        split = data.get_split()

    # DTI
    def test_DTI(self):
        from tdc.multi_pred import DTI
        data = DTI(name='DAVIS')
        split = data.get_split()

    # DDI
    def test_DDI(self):
        from tdc.multi_pred import DDI
        data = DDI(name='DrugBank')
        split = data.get_split()
        from tdc.utils import get_label_map
        get_label_map(name='DrugBank', task='DDI')

    # PPI
    def test_PPI(self):
        from tdc.multi_pred import PPI
        data = PPI(name='HuRI')
        split = data.get_split()
        data = data.neg_sample(frac=1)

    # GDA
    def test_GDA(self):
        from tdc.multi_pred import GDA
        data = GDA(name='DisGeNET')
        split = data.get_split()

    # DrugRes
    def test_DrugRes(self):
        from tdc.multi_pred import DrugRes
        data = DrugRes(name='GDSC1')
        split = data.get_split()

    # DrugSyn
    def test_DrugSyn(self):
        from tdc.multi_pred import DrugSyn
        data = DrugSyn(name='OncoPolyPharmacology')
        split = data.get_split()

    # PeptideMHC
    def test_PeptideMHC(self):
        from tdc.multi_pred import PeptideMHC
        data = PeptideMHC(name='MHC1_IEDB-IMGT_Nielsen')
        split = data.get_split()

    # AntibodyAff
    def test_AntibodyAff(self):
        from tdc.multi_pred import AntibodyAff
        data = AntibodyAff(name='Protein_SAbDab')
        split = data.get_split()

    # MTI
    def test_MTI(self):
        from tdc.multi_pred import MTI
        data = MTI(name='miRTarBase')
        split = data.get_split()

    # Catalyst
    def test_Catalyst(self):
        from tdc.multi_pred import Catalyst
        data = Catalyst(name='USPTO_Catalyst')
        split = data.get_split()

    # MolGen
    def test_MolGen(self):
        from tdc.generation import MolGen
        data = MolGen(name='ZINC')
        split = data.get_split()

    # PairMolGen
    def test_PairMolGen(self):
        from tdc.generation import PairMolGen
        data = PairMolGen(name='DRD2')
        split = data.get_split()

    # RetroSyn Reaction
    def test_RetroSyn(self):
        from tdc.generation import RetroSyn
        data = RetroSyn(name='USPTO-50K')
        split = data.get_split()

    def tearDown(self):
        print(os.getcwd())
        shutil.rmtree(os.path.join(os.getcwd(), "data"))
