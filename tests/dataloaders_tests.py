# data loader tests

# ADME
from tdc.single_pred import ADME
data = ADME(name = 'Caco2_Wang')
split = data.get_split()

# Tox
from tdc.utils import retrieve_label_name_list
label_list = retrieve_label_name_list('Tox21')
from tdc.single_pred import Tox
data = Tox(name = 'Tox21', label_name = label_list[0])
split = data.get_split()

# HTS
from tdc.single_pred import HTS
data = HTS(name = 'SARSCoV2_Vitro_Touret')
split = data.get_split()

# QM
from tdc.utils import retrieve_label_name_list
label_list = retrieve_label_name_list('QM7b')
from tdc.single_pred import QM
data = QM(name = 'QM7b', label_name = label_list[0])
split = data.get_split()

# Yields
from tdc.single_pred import Yields
data = Yields(name = 'Buchwald-Hartwig')
split = data.get_split()

# Paratope
from tdc.single_pred import Paratope
data = Paratope(name = 'SAbDab_Liberis')
split = data.get_split()

# Epitope
from tdc.single_pred import Epitope
data = Epitope(name = 'IEDB_Jespersen')
split = data.get_split()

# Develop
from tdc.utils import retrieve_label_name_list
label_list = retrieve_label_name_list('TAP')

from tdc.single_pred import Develop
data = Develop(name = 'TAP', label_name = label_list[0])
split = data.get_split()

# DTI
from tdc.multi_pred import DTI
data = DTI(name = 'DAVIS')
split = data.get_split()

# DDI
from tdc.multi_pred import DDI
data = DDI(name = 'DrugBank')
split = data.get_split()
from tdc.utils import get_label_map
get_label_map(name = 'DrugBank', task = 'DDI')

# PPI
from tdc.multi_pred import PPI
data = PPI(name = 'HuRI')
split = data.get_split()
data = data.neg_sample(frac = 1)

# GDA
from tdc.multi_pred import GDA
data = GDA(name = 'DisGeNET')
split = data.get_split()

# DrugRes
from tdc.multi_pred import DrugRes
data = DrugRes(name = 'GDSC1')
split = data.get_split()

# DrugSyn
from tdc.multi_pred import DrugSyn
data = DrugSyn(name = 'OncoPolyPharmacology')
split = data.get_split()

# PeptideMHC
from tdc.multi_pred import PeptideMHC
data = PeptideMHC(name = 'MHC1_IEDB-IMGT_Nielsen')
split = data.get_split()

# AntibodyAff
from tdc.multi_pred import AntibodyAff
data = AntibodyAff(name = 'Protein_SAbDab')
split = data.get_split()

# MTI
from tdc.multi_pred import MTI
data = MTI(name = 'miRTarBase')
split = data.get_split()

# Catalyst
from tdc.multi_pred import Catalyst
data = Catalyst(name = 'USPTO_Catalyst')
split = data.get_split()

# MolGen
from tdc.generation import MolGen
data = MolGen(name = 'ZINC')
split = data.get_split()

# PairMolGen
from tdc.generation import PairMolGen
data = PairMolGen(name = 'DRD2')
split = data.get_split()

# RetroSyn
from tdc.generation import RetroSyn
data = RetroSyn(name = 'USPTO-50K')
split = data.get_split()

# Reaction
from tdc.generation import Reaction
data = Reaction(name = 'USPTO')
split = data.get_split()
