import pickle 
import numpy as np 
import re
import os.path as op
import math
from collections import defaultdict, Iterable
from abc import abstractmethod
from functools import partial
from typing import List
import time
import os 

try:
	from sklearn import svm
	# from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score
except:
	ImportError("Please install sklearn by 'conda install -c anaconda scikit-learn' or 'pip install scikit-learn '! ")

try: 
  import rdkit
  from rdkit import Chem, DataStructs
  from rdkit.Chem import AllChem
  from rdkit.Chem import Descriptors
  import rdkit.Chem.QED as QED
  from rdkit import rdBase
  rdBase.DisableLog('rdApp.error')
  from rdkit.Chem import rdMolDescriptors
  from rdkit.six.moves import cPickle
  from rdkit.six import iteritems
  from rdkit.Chem.Fingerprints import FingerprintMols
  from rdkit.Chem import MACCSkeys
except:
  raise ImportError("Please install rdkit by 'conda install -c conda-forge rdkit'! ")	

try:
	from scipy.stats.mstats import gmean
except:
	raise ImportError("Please install rdkit by 'pip install scipy'! ") 


try:
	import networkx as nx 
except:
	raise ImportError("Please install networkx by 'pip install networkx'! ")	

from tdc.utils import oracle_load, print_sys, install

# from _smiles2pubchem import smiles2pubchem


def smiles_to_rdkit_mol(smiles):
	mol = Chem.MolFromSmiles(smiles)
	#  Sanitization check (detects invalid valence)
	if mol is not None:
		try:
			Chem.SanitizeMol(mol)
		except ValueError:
			return None
	return mol

def smiles_2_fingerprint_ECFP4(smiles):
	molecule = smiles_to_rdkit_mol(smiles)
	fp = AllChem.GetMorganFingerprint(molecule, 2)
	return fp 

def smiles_2_fingerprint_FCFP4(smiles):
	molecule = smiles_to_rdkit_mol(smiles)
	fp = AllChem.GetMorganFingerprint(molecule, 2, useFeatures=True)
	return fp 

def smiles_2_fingerprint_AP(smiles):
	molecule = smiles_to_rdkit_mol(smiles)
	fp = AllChem.GetAtomPairFingerprint(molecule, maxLength=10)
	return fp 

def smiles_2_fingerprint_ECFP6(smiles):
	molecule = smiles_to_rdkit_mol(smiles)
	fp = AllChem.GetMorganFingerprint(molecule, 3)
	return fp 



def smiles_to_rdkit_mol(smiles):
  mol = Chem.MolFromSmiles(smiles)
  #  Sanitization check (detects invalid valence)
  if mol is not None:
    try:
      Chem.SanitizeMol(mol)
    except ValueError:
      return None
  return mol





def smiles2morgan(s, radius = 2, nBits = 1024):
    try:
        s = canonicalize(s)
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print_sys('rdkit not found this smiles for morgan: ' + s + ' convert to all 0 features')
        features = np.zeros((nBits, ))
    return features

def smiles2rdkit2d(s): 
    s = canonicalize(s)
    try:
        from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
    except:
        raise ImportError("Please install pip install git+https://github.com/bp-kelley/descriptastorus and pip install pandas-flavor")   
    try:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = np.array(generator.process(s)[1:])
        NaNs = np.isnan(features)
        features[NaNs] = 0
    except:
        print_sys('descriptastorus not found this smiles: ' + s + ' convert to all 0 features')
        features = np.zeros((200, ))
    return np.array(features)

def smiles2daylight(s):
  try:
    s = canonicalize(s)
    NumFinger = 2048
    mol = Chem.MolFromSmiles(s)
    bv = FingerprintMols.FingerprintMol(mol)
    temp = tuple(bv.GetOnBits())
    features = np.zeros((NumFinger, ))
    features[np.array(temp)] = 1
  except:
    print_sys('rdkit not found this smiles: ' + s + ' convert to all 0 features')
    features = np.zeros((2048, ))
  return np.array(features)

def smiles2maccs(s):
  s = canonicalize(s)
  mol = Chem.MolFromSmiles(s)
  fp = MACCSkeys.GenMACCSKeys(mol)
  arr = np.zeros((0,), dtype=np.float64)
  DataStructs.ConvertToNumpyArray(fp,arr)
  return arr

'''
  ECFP2 ---- 1
  ECFP4 ---- 2
  ECFP6 ---- 3
  xxxxxxxxx  ------  https://github.com/rdkit/benchmarking_platform/blob/master/scoring/fingerprint_lib.py 

'''
def smiles2ECFP2(smiles):
  nbits = 2048
  smiles = canonicalize(smiles)
  molecule = smiles_to_rdkit_mol(smiles)
  fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 1, nBits=nbits)
  arr = np.zeros((0,), dtype=np.float64)
  DataStructs.ConvertToNumpyArray(fp,arr)
  return arr 


def smiles2ECFP4(smiles):
  nbits = 2048
  smiles = canonicalize(smiles)
  molecule = smiles_to_rdkit_mol(smiles)
  fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=nbits)
  arr = np.zeros((0,), dtype=np.float64)
  DataStructs.ConvertToNumpyArray(fp,arr)
  return arr 


def smiles2ECFP6(smiles):
  nbits = 2048
  smiles = canonicalize(smiles)
  molecule = smiles_to_rdkit_mol(smiles)
  fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 1, nBits=nbits)
  arr = np.zeros((0,), dtype=np.float64)
  DataStructs.ConvertToNumpyArray(fp,arr)
  return arr 

# def smiles2smart(smiles):


class MoleculeFingerprint:

    '''
    Example:
    MolFP = MoleculeFingerprint(fp = 'ECFP6')
    out = MolFp('Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC')
    # np.array([1, 0, 1, .....])
    out = MolFp(['Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC',
                'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C'])
    # np.array([[1, 0, 1, .....],
                [0, 0, 1, .....]])
    
    Supporting FPs:
    Basic_Descriptors(atoms, chirality, ....), ECFP2, ECFP4, ECFP6, MACCS, Daylight-type, RDKit2D, Morgan, PubChem
    '''

    def __init__(self, fp = 'ECFP4'):
        fp2func = {'ECFP2': smiles2ECFP2, 
               'ECFP4': smiles2ECFP4, 
               'ECFP6': smiles2ECFP6, 
               'MACCS': smiles2maccs, 
               'Daylight': smiles2daylight, 
               'RDKit2D': smiles2rdkit2d, 
               'Morgan': smiles2morgan, 
               'PubChem': smiles2pubchem}
        try:
            assert fp in fp2func
        except:
            raise Exception("The fingerprint you specify are not supported. \
              It can only among 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'")

        self.fp = fp
        self.func = fp2func[fp]

    def __call__(self, x):
        if type(x)==str:
          return self.func(x)
        elif type(x)==list:
          lst = list(map(self.func, x))
          arr = np.vstack(lst)
          return arr 

def smiles2selfies(smiles):
  smiles = canonicalize(smiles)
  return sf.encoder(smiles)

def selfies2smiles(selfies):
  return canonicalize(sf.decoder(selfies))


def smiles2mol(smiles):
    smiles = canonicalize(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol 

def bondtype2idx(bond_type):
  if bond_type == Chem.rdchem.BondType.SINGLE:
    return 1
  elif bond_type == Chem.rdchem.BondType.DOUBLE:
    return 2
  elif bond_type == Chem.rdchem.BondType.TRIPLE:
    return 3
  elif bond_type == Chem.rdchem.BondType.AROMATIC:
    return 4

def smiles2graph2D(smiles):
  smiles = canonicalize(smiles)
  mol = smiles2mol(smiles)
  n_atoms = mol.GetNumAtoms()
  idx2atom = {atom.GetIdx():atom.GetSymbol() for atom in mol.GetAtoms()}
  adj_matrix = np.zeros((n_atoms, n_atoms), dtype = int)
  for bond in mol.GetBonds():
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    idx1 = a1.GetIdx()
    idx2 = a2.GetIdx() 
    bond_type = bond.GetBondType()
    bond_idx = bondtype2idx(bond_type)
    adj_matrix[idx1,idx2] = bond_idx
    adj_matrix[idx2,idx1] = bond_idx
  return idx2atom, adj_matrix

# def selfies2graph2D(selfies):
#   smiles = selfies2smiles(selfies)
#   smiles = canonicalize(smiles)
#   return smiles2graph2D(smiles)


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

############### PyG begin ###############
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
# https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/chemutils.py

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def smiles2PyG(smiles):
  smiles = canonicalize(smiles)
  mol = Chem.MolFromSmiles(smiles)
  n_atoms = mol.GetNumAtoms()
  atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
  atom_features = torch.stack(atom_features)
  y = [atom.GetSymbol() for atom in mol.GetAtoms()]
  y = list(map(lambda x: ELEM_LIST.index(x) if x in ELEM_LIST else len(ELEM_LIST)-1 , y))
  y = torch.LongTensor(y)
  bond_features = []
  for bond in mol.GetBonds():
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    idx1 = a1.GetIdx()
    idx2 = a2.GetIdx() 
    bond_features.extend([[idx1, idx2], [idx2, idx1]])
  bond_features = torch.LongTensor(bond_features)

  data = Data(x=atom_features, edge_index=bond_features.T)

  return data

# def selfies2PyG(selfies):
#   smiles = selfies2smiles(selfies)
#   smiles = canonicalize(smiles)
#   return smiles2PyG(smiles)

def molfile2PyG(molfile):
  smiles = molfile2smiles(molfile)
  smiles = canonicalize(smiles)
  return smiles2PyG(smiles)
############### PyG end ###############

############### DGL begin ###############
def smiles2DGL(smiles):
  smiles = canonicalize(smiles)
  mol = Chem.MolFromSmiles(smiles)
  n_atoms = mol.GetNumAtoms()
  bond_features = []
  for bond in mol.GetBonds():
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    idx1 = a1.GetIdx()
    idx2 = a2.GetIdx() 
    bond_features.extend([[idx1, idx2], [idx2, idx1]])
  src, dst = tuple(zip(*bond_features))
  g = dgl.DGLGraph()
  g.add_nodes(n_atoms)
  g.add_edges(src, dst) 
  return g 

# def selfies2DGL(selfies):
#   smiles = selfies2smiles(selfies)
#   smiles = canonicalize(smiles)
#   return smiles2DGL(smiles)

# def molfile2DGL(molfile):
#   smiles = molfile2smiles(molfile)
#   smiles = canonicalize(smiles)
#   return smiles2DGL(smiles)


############### DGL end ###############

############## begin xyz2mol ################
global __ATOM_LIST__
__ATOM_LIST__ = \
    ['h',  'he',
     'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne',
     'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',
     'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',
     'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
     'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
     'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe',
     'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
     'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt',
     'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
     'fr', 'ra', 'ac', 'th', 'pa', 'u',  'np', 'pu']


global atomic_valence
global atomic_valence_electrons

atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[5] = [3,4]
atomic_valence[6] = [4]
atomic_valence[7] = [3,4]
atomic_valence[8] = [2,1,3]
atomic_valence[9] = [1]
atomic_valence[14] = [4]
atomic_valence[15] = [5,3] #[5,4,3]
atomic_valence[16] = [6,3,2] #[6,4,2]
atomic_valence[17] = [1]
atomic_valence[32] = [4]
atomic_valence[35] = [1]
atomic_valence[53] = [1]

atomic_valence_electrons = {}
atomic_valence_electrons[1] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[32] = 4
atomic_valence_electrons[35] = 7
atomic_valence_electrons[53] = 7


def str_atom(atom):
    """
    convert integer atom to string atom
    """
    global __ATOM_LIST__
    atom = __ATOM_LIST__[atom - 1]
    return atom


def int_atom(atom):
    """
    convert str atom to integer atom
    """
    global __ATOM_LIST__
    #print(atom)
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1


def get_UA(maxValence_list, valence_list):
    """
    """
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if not maxValence - valence > 0:
            continue
        UA.append(i)
        DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, use_graph=True):
    """
    """
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = get_UA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA, AC, use_graph=use_graph)[0]

    return BO


def valences_not_too_large(BO, valences):
    """
    """
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False

    return True

def charge_is_OK(BO, AC, charge, DU, atomic_valence_electrons, atoms, valences,
                 allow_charged_fragments=True):
    # total charge
    Q = 0

    # charge fragment list
    q_list = []

    if allow_charged_fragments:

        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atoms):
            q = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1

            if q != 0:
                q_list.append(q)

    return (charge == Q)

def BO_is_OK(BO, AC, charge, DU, atomic_valence_electrons, atoms, valences,
    allow_charged_fragments=True):
    """
    Sanity of bond-orders
    args:
        BO -
        AC -
        charge -
        DU - 
    optional
        allow_charges_fragments - 
    returns:
        boolean - true of molecule is OK, false if not
    """

    if not valences_not_too_large(BO, valences):
        return False

    check_sum = (BO - AC).sum() == sum(DU)
    check_charge = charge_is_OK(BO, AC, charge, DU, atomic_valence_electrons, atoms, valences,
                                allow_charged_fragments)

    if check_charge and check_sum: 
        return True

    return False


def get_atomic_charge(atom, atomic_valence_electrons, BO_valence):
    """
    """

    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif atom == 15 and BO_valence == 5:
        charge = 0
    elif atom == 16 and BO_valence == 6:
        charge = 0
    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge


def clean_charges(mol):
    """
    This hack should not be needed anymore, but is kept just in case
    """

    Chem.SanitizeMol(mol)
    #rxn_smarts = ['[N+:1]=[*:2]-[C-:3]>>[N+0:1]-[*:2]=[C-0:3]',
    #              '[N+:1]=[*:2]-[O-:3]>>[N+0:1]-[*:2]=[O-0:3]',
    #              '[N+:1]=[*:2]-[*:3]=[*:4]-[O-:5]>>[N+0:1]-[*:2]=[*:3]-[*:4]=[O-0:5]',
    #              '[#8:1]=[#6:2]([!-:6])[*:3]=[*:4][#6-:5]>>[*-:1][*:2]([*:6])=[*:3][*:4]=[*+0:5]',
    #              '[O:1]=[c:2][c-:3]>>[*-:1][*:2][*+0:3]',
    #              '[O:1]=[C:2][C-:3]>>[*-:1][*:2]=[*+0:3]']

    rxn_smarts = ['[#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][CX3-,NX3-:5][#6,#7:6]1=[#6,#7:7]>>'
                  '[#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][-0,-0:5]=[#6,#7:6]1[#6-,#7-:7]',
                  '[#6,#7:1]1=[#6,#7:2][#6,#7:3](=[#6,#7:4])[#6,#7:5]=[#6,#7:6][CX3-,NX3-:7]1>>'
                  '[#6,#7:1]1=[#6,#7:2][#6,#7:3]([#6-,#7-:4])=[#6,#7:5][#6,#7:6]=[-0,-0:7]1']

    fragments = Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=False)

    for i, fragment in enumerate(fragments):
        for smarts in rxn_smarts:
            patt = Chem.MolFromSmarts(smarts.split(">>")[0])
            while fragment.HasSubstructMatch(patt):
                rxn = AllChem.ReactionFromSmarts(smarts)
                ps = rxn.RunReactants((fragment,))
                fragment = ps[0][0]
                Chem.SanitizeMol(fragment)
        if i == 0:
            mol = fragment
        else:
            mol = Chem.CombineMols(mol, fragment)

    return mol


def BO2mol(mol, BO_matrix, atoms, atomic_valence_electrons,
           mol_charge, allow_charged_fragments=True):
    """
    based on code written by Paolo Toscani
    From bond order, atoms, valence structure and total charge, generate an
    rdkit molecule.
    args:
        mol - rdkit molecule
        BO_matrix - bond order matrix of molecule
        atoms - list of integer atomic symbols
        atomic_valence_electrons -
        mol_charge - total charge of molecule
    optional:
        allow_charged_fragments - bool - allow charged fragments
    returns
        mol - updated rdkit molecule with bond connectivity
    """

    l = len(BO_matrix)
    l2 = len(atoms)
    BO_valences = list(BO_matrix.sum(axis=1))

    if (l != l2):
        raise RuntimeError('sizes of adjMat ({0:d}) and Atoms {1:d} differ'.format(l, l2))

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }

    for i in range(l):
        for j in range(i + 1, l):
            bo = int(round(BO_matrix[i, j]))
            if (bo == 0):
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    if allow_charged_fragments:
        mol = set_atomic_charges(
            mol,
            atoms,
            atomic_valence_electrons,
            BO_valences,
            BO_matrix,
            mol_charge)
    else:
        mol = set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences)

    return mol


def set_atomic_charges(mol, atoms, atomic_valence_electrons,
                       BO_valences, BO_matrix, mol_charge):
    """
    """
    q = 0
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                q += 1
                charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1

        if (abs(charge) > 0):
            a.SetFormalCharge(int(charge))

    #mol = clean_charges(mol)

    return mol


def set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences):
    """
    The number of radical electrons = absolute atomic charge
    """
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(
            atom,
            atomic_valence_electrons[atom],
            BO_valences[i])

        if (abs(charge) > 0):
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol


def get_bonds(UA, AC):
    """
    """
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k + 1:]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))

    return bonds


def get_UA_pairs(UA, AC, use_graph=True):
    """
    """

    bonds = get_bonds(UA, AC)

    if len(bonds) == 0:
        return [()]

    if use_graph:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]

        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs


def AC2BO(AC, atoms, charge, allow_charged_fragments=True, use_graph=True):
    """
    implemenation of algorithm shown in Figure 2
    UA: unsaturated atoms
    DU: degree of unsaturation (u matrix in Figure)
    best_BO: Bcurr in Figure
    """

    global atomic_valence
    global atomic_valence_electrons

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    AC_valence = list(AC.sum(axis=1))
    
    for i,(atomicNum,valence) in enumerate(zip(atoms,AC_valence)):
        # valence can't be smaller than number of neighbourgs
        possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]
        if not possible_valence:
            print_sys('Valence of atom',i,'is',valence,'which bigger than allowed max',max(atomic_valence[atomicNum]),'. Stopping')
            sys.exit()
        valences_list_of_lists.append(possible_valence)

    # convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = itertools.product(*valences_list_of_lists)

    best_BO = AC.copy()

    for valences in valences_list:

        UA, DU_from_AC = get_UA(valences, AC_valence)

        check_len = (len(UA) == 0)
        if check_len:
            check_bo = BO_is_OK(AC, AC, charge, DU_from_AC,
                atomic_valence_electrons, atoms, valences,
                allow_charged_fragments=allow_charged_fragments)
        else:
            check_bo = None

        if check_len and check_bo:
            return AC, atomic_valence_electrons

        UA_pairs_list = get_UA_pairs(UA, AC, use_graph=use_graph)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, use_graph=use_graph)
            status = BO_is_OK(BO, AC, charge, DU_from_AC,
                        atomic_valence_electrons, atoms, valences,
                        allow_charged_fragments=allow_charged_fragments)
            charge_OK = charge_is_OK(BO, AC, charge, DU_from_AC, atomic_valence_electrons, atoms, valences,
                                     allow_charged_fragments=allow_charged_fragments)

            if status:
                return BO, atomic_valence_electrons
            elif BO.sum() >= best_BO.sum() and valences_not_too_large(BO, valences) and charge_OK:
                best_BO = BO.copy()

    return best_BO, atomic_valence_electrons


def AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, use_graph=True):
    """
    """

    # convert AC matrix to bond order (BO) matrix
    BO, atomic_valence_electrons = AC2BO(
        AC,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph)


    # add BO connectivity and charge info to mol object
    mol = BO2mol(
        mol,
        BO,
        atoms,
        atomic_valence_electrons,
        charge,
        allow_charged_fragments=allow_charged_fragments)
    # print('mol', mol)
    # If charge is not correct don't return mol
    # if Chem.GetFormalCharge(mol) != charge:
    #     return []

    # BO2mol returns an arbitrary resonance form. Let's make the rest
    mols = rdchem.ResonanceMolSupplier(mol, Chem.UNCONSTRAINED_CATIONS, Chem.UNCONSTRAINED_ANIONS)
    mols = [mol for mol in mols]

    return mols, BO 


def get_proto_mol(atoms):
    """
    """
    mol = Chem.MolFromSmarts("[#" + str(atoms[0]) + "]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atoms)):
        a = Chem.Atom(atoms[i])
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def read_xyz_file(filename, look_for_charge=True):
    """
    """

    atomic_symbols = []
    xyz_coordinates = []
    charge = 0
    title = ""

    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                title = line
                if "charge=" in line:
                    charge = int(line.split("=")[1])
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, charge, xyz_coordinates


def xyz2AC(atoms, xyz, charge, use_huckel=False):
    """
    atoms and coordinates to atom connectivity (AC)
    args:
        atoms - int atom types
        xyz - coordinates
        charge - molecule charge
    optional:
        use_huckel - Use Huckel method for atom connecitivty
    returns
        ac - atom connectivity matrix
        mol - rdkit molecule
    """

    if use_huckel:
        return xyz2AC_huckel(atoms, xyz, charge)
    else:
        return xyz2AC_vdW(atoms, xyz)


def xyz2AC_vdW(atoms, xyz):

    # Get mol template
    mol = get_proto_mol(atoms)

    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    AC = get_AC(mol)

    return AC, mol


def get_AC(mol, covalent_factor=1.3):
    """
    Generate adjacent matrix from atoms and coordinates.
    AC is a (num_atoms, num_atoms) matrix with 1 being covalent bond and 0 is not
    covalent_factor - 1.3 is an arbitrary factor
    args:
        mol - rdkit molobj with 3D conformer
    optional
        covalent_factor - increase covalent bond length threshold with facto
    returns:
        AC - adjacent matrix
    """

    # Calculate distance matrix
    dMat = Chem.Get3DDistanceMatrix(mol)

    pt = Chem.GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    AC = np.zeros((num_atoms, num_atoms), dtype=int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * covalent_factor
        for j in range(i + 1, num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * covalent_factor
            if dMat[i, j] <= Rcov_i + Rcov_j:
                AC[i, j] = 1
                AC[j, i] = 1

    return AC


def xyz2AC_huckel(atomicNumList,xyz,charge):
    """
    args
        atomicNumList - atom type list
        xyz - coordinates
        charge - molecule charge
    returns
        ac - atom connectivity
        mol - rdkit molecule
    """
    mol = get_proto_mol(atomicNumList)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i,(xyz[i][0],xyz[i][1],xyz[i][2]))
    mol.AddConformer(conf)

    num_atoms = len(atomicNumList)
    AC = np.zeros((num_atoms,num_atoms)).astype(int)

    mol_huckel = Chem.Mol(mol)
    mol_huckel.GetAtomWithIdx(0).SetFormalCharge(charge) #mol charge arbitrarily added to 1st atom    

    passed,result = rdEHTTools.RunMol(mol_huckel)
    opop = result.GetReducedOverlapPopulationMatrix()
    tri = np.zeros((num_atoms, num_atoms))
    tri[np.tril(np.ones((num_atoms, num_atoms), dtype=bool))] = opop #lower triangular to square matrix
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            pair_pop = abs(tri[j,i])   
            if pair_pop >= 0.15: #arbitry cutoff for bond. May need adjustment
                AC[i,j] = 1
                AC[j,i] = 1

    return AC, mol


def chiral_stereo_check(mol):
    """
    Find and embed chiral information into the model based on the coordinates
    args:
        mol - rdkit molecule, with embeded conformer
    """
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)

    return


def xyz2mol(atoms, coordinates,
    charge=0,
    allow_charged_fragments=True,
    use_graph=True,
    use_huckel=False,
    embed_chiral=True):
    """
    Generate a rdkit molobj from atoms, coordinates and a total_charge.
    args:
        atoms - list of atom types (int)
        coordinates - 3xN Cartesian coordinates
        charge - total charge of the system (default: 0)
    optional:
        allow_charged_fragments - alternatively radicals are made
        use_graph - use graph (networkx)
        use_huckel - Use Huckel method for atom connectivity prediction
        embed_chiral - embed chiral information to the molecule
    returns:
        mols - list of rdkit molobjects
    """

    # Get atom connectivity (AC) matrix, list of atomic numbers, molecular charge,
    # and mol object with no connectivity information
    AC, mol = xyz2AC(atoms, coordinates, charge, use_huckel=use_huckel)
    # print('AC, mol', AC, mol)
    # Convert AC to bond order matrix and add connectivity and charge info to
    # mol object
    new_mols, BO = AC2mol(mol, AC, atoms, charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph)
    # Check for stereocenters and chiral centers
    # print('new_mols', new_mols)
    if embed_chiral:
        for new_mol in new_mols:
            chiral_stereo_check(new_mol)
    return new_mols, BO 

def xyzfile2mol(xyzfile):
  atoms, charge, xyz_coordinates = read_xyz_file(xyzfile)
  # print(atoms, xyz_coordinates)
  charged_fragments = True 
  # quick is faster for large systems but requires networkx
  # if you don't want to install networkx set quick=False and
  # uncomment 'import networkx as nx' at the top of the file
  quick = True 

  # chiral comment
  embed_chiral = True ### random 

  # huckel uses extended Huckel bond orders to locate bonds (requires RDKit 2019.9.1 or later)
  # otherwise van der Waals radii are used
  use_huckel = True 

  # if explicit charge from args, set it
  charge = charge 

  # Get the molobjs
  mols, BO = xyz2mol(atoms, xyz_coordinates,
        charge=charge,
        use_graph=quick,
        allow_charged_fragments=charged_fragments,
        embed_chiral=embed_chiral,
        use_huckel=use_huckel)
  return mols[0], BO

def mol2smiles(mol):
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles

def xyzfile2smiles(xyzfile):
  mol, _ = xyzfile2mol(xyzfile)
  smiles = mol2smiles(mol)
  smiles = canonicalize(smiles)
  return smiles 

def xyzfile2selfies(xyzfile):
  smiles = xyzfile2smiles(xyzfile)
  smiles = canonicalize(smiles)
  selfies = smiles2selfies(smiles)
  return selfies 

def distance3d(coordinate_1, coordinate_2):
  return np.sqrt(sum([(c1-c2)**2 for c1,c2 in zip(coordinate_1, coordinate_2)])) 

def upper_atom(atomsymbol):
  return atomsymbol[0].upper() + atomsymbol[1:]

def xyzfile2graph3d(xyzfile):
  atoms, charge, xyz_coordinates = read_xyz_file(file)
  num_atoms = len(atoms)
  distance_adj_matrix = np.zeros((num_atoms, num_atoms))
  for i in range(num_atoms):
    for j in range(i+1, num_atoms):
      distance = distance3d(xyz_coordinates[i], xyz_coordinates[j])
      distance_adj_matrix[i,j] = distance_adj_matrix[j,i] = distance 
  idx2atom = {idx:upper_atom(str_atom(atom)) for idx,atom in enumerate(atoms)}
  mol, BO = xyzfile2mol(xyzfile)
  return idx2atom, distance_adj_matrix, BO 
############## end xyz2mol ################

def sdffile2smiles_lst(sdffile):
  from rdkit.Chem.PandasTools import LoadSDF
  df = LoadSDF(sdffile, smilesName='SMILES')
  smiles_lst = df['SMILES'].to_list() 
  return smiles_lst
 

def sdffile2mol_conformer(sdffile):
  from rdkit.Chem.PandasTools import LoadSDF
  df = LoadSDF(sdffile, smilesName='SMILES')
  mol_lst = df['ROMol'].tolist() 
  conformer_lst = []
  for mol in mol_lst:
    conformer = mol.GetConformer(id=0)
    conformer_lst.append(conformer)
  mol_conformer_lst = list(zip(mol_lst, conformer_lst))
  return mol_conformer_lst   



def mol_conformer2graph3d(mol_conformer_lst):
  graph3d_lst = []
  bond2num = {'SINGLE': 1, 'DOUBLE':2, 'TRIPLE':3, "AROMATIC":4}
  for mol, conformer in mol_conformer_lst:
    atom_num = mol.GetNumAtoms() 
    distance_adj_matrix = np.zeros((atom_num, atom_num))
    bondtype_adj_matrix = np.zeros((atom_num, atom_num), dtype = int)
    idx2atom = {i:v.GetSymbol() for i,v in enumerate(mol.GetAtoms())}
    positions = []
    for i in range(atom_num):
      pos = conformer.GetAtomPosition(i)
      coordinate = np.array([pos.x, pos.y, pos.z]).reshape(1,3)
      positions.append(coordinate)
    positions = np.concatenate(positions, 0)
    for i in range(atom_num):
      for j in range(i+1, atom_num):
        distance_adj_matrix[i,j] = distance_adj_matrix[j,i] = distance3d(positions[i], positions[j])
    for bond in mol.GetBonds():
      a1 = bond.GetBeginAtom().GetIdx()
      a2 = bond.GetEndAtom().GetIdx()
      bt = bond.GetBondType()
      bondtype_adj_matrix[a1,a2] = bond2num[str(bt)]
      bondtype_adj_matrix[a1,a2] = bond2num[str(bt)]
    graph3d_lst.append((idx2atom, distance_adj_matrix, bondtype_adj_matrix))
  return graph3d_lst


def sdffile2graph3d_lst(sdffile):
  mol_conformer_lst = sdffile2mol_conformer(sdffile)
  graph3d_lst = mol_conformer2graph3d(mol_conformer_lst)
  return graph3d_lst




def sdffile2selfies_lst(sdf):
  smiles_lst = sdffile2smiles_lst(sdf)
  selfies_lst = list(map(smiles2selfies, smiles_lst))
  return selfies_lst 




def smiles_lst2coulomb(smiles_lst):
  molecules = [Molecule(smiles, 'smiles') for smiles in smiles_lst]
  for mol in molecules:   
    mol.to_xyz(optimizer='UFF')
  cm = CoulombMatrix(cm_type='UM', n_jobs=-1)
  features = cm.represent(molecules)
  features = features.to_numpy() 
  return features 
  ## (nmol, max_atom_n**2),
  ## where max_atom_n is maximal number of atom in the smiles_lst 
  ## features[i].reshape(max_atom_n, max_atom_n)[:3,:3]  -> 3*3 Coulomb matrix   

def sdffile2coulomb(sdf):
  smiles_lst = sdffile2smiles_lst(sdf)
  return smiles_lst2coulomb(smiles_lst)

def xyzfile2coulomb(xyzfile):
  smiles = xyzfile2smiles(xyzfile)
  smiles = canonicalize(smiles)
  return smiles_lst2coulomb([smiles])




#2D_format = ['SMILES', 'SELFIES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem']
#3D_format = ['Graph3D', 'Coulumb']

## XXX2smiles
def molfile2smiles(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles 


def mol2file2smiles(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles 


def smiles2smiles(smiles):
	return canonicalize(smiles)

## smiles2xxx 

convert_dict = {
          'SMILES': ['SELFIES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'],
          'SELFIES': ['SMILES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'], 
          'mol': ['SMILES', 'SELFIES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'],
          'mol2': ['SMILES', 'SELFIES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'], 
          'SDF': ['SMILES', 'SELFIES', 'Graph3D', 'Coulumb'],
          'XYZ': ['SMILES', 'SELFIES', 'Graph3D', 'Coulumb'],  
        }

fingerprints_list = ['ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem']

twoD_format = ['SMILES', 'SELFIES', 'mol', 'mol2', ]
threeD_format = ['SDF', 'XYZ', ]


class MolConvert:

    '''
    Example:
        convert = MolConvert(src = ‘SMILES’, dst = ‘Graph2D’)
        g = convert(‘Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC’)
        # g: graph with edge, node features
        g = convert(['Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC',
                  'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C'])
        # g: a list of graphs with edge, node features
        if src is 2D, dst can be only 2D output
        if src is 3D, dst can be both 2D and 3D outputs
        src: 2D - [SMILES, SELFIES]
              3D - [SDF file, XYZ file] 
        dst: 2D - [2D Graph (+ PyG, DGL format), Canonical SMILES, SELFIES, Fingerprints] 
              3D - [3D graphs (adj matrix entry is (distance, bond type)), Coulumb Matrix] 
    '''

    def __init__(self, src = 'SMILES', dst = 'Graph2D', radius = 2, nBits = 1024):
        self._src = src
        self._dst = dst
        self._radius = radius 
        self._nbits = nBits

        self.convert_dict = convert_dict
        if 'SELFIES' == src or 'SELFIES' == dst:
          try:
            import selfies as sf
            global sf 
          except:
            raise Exception("Please install selfies via 'pip install selfies'")

        if 'Coulumb' == dst:
          try:
            from chemml.chem import CoulombMatrix, Molecule
            global CoulombMatrix, Molecule 
          except:
            raise Exception("Please install chemml via 'pip install pybel' and 'pip install chemml'. ")

        if 'PyG' == dst:
          try:
            import torch
            from torch_geometric.data import Data
            global torch 
            global Data
          except:
            raise Exception("Please install PyTorch Geometric via 'https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html'.")

        if 'DGL' == dst:
          try: 
            import dgl
            global dgl 
          except:
            raise Exception("Please install DGL via 'pip install dgl'.")

        try:
          assert src in self.convert_dict
        except:
          raise Exception("src format is not supported")
        try:
          assert dst in self.convert_dict[src]
        except:
          raise Exception('It is not supported to convert src to dst.')


        if src in twoD_format:
            ### 1. src -> SMILES 
            if src == "SMILES":
                f1 = canonicalize 
            elif src == "SELFIES":
                f1 = selfies2smiles 
            elif src == "mol":
                f1 = molfile2smiles 
            elif src == "mol2":
                f1 = mol2file2smiles 

            ### 2. SMILES -> all 
            # 'SMILES', 'SELFIES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'
            if dst == 'SMILES':
                f2 = canonicalize 
            elif dst == 'SELFIES':
                f2 = smiles2selfies 
            elif dst == "Graph2D":
                f2 = smiles2graph2D 
            elif dst == "PyG":
                f2 = smiles2PyG 
            elif dst == "DGL":
                f2 = smiles2DGL
            elif dst == "ECFP2":
                f2 = smiles2ECFP2 
            elif dst == "ECFP4":
                f2 = smiles2ECFP4 
            elif dst == "MACCS":
                f2 = smiles2maccs 
            elif dst == "Daylight":
                f2 = smiles2daylight 
            elif dst == "RDKit2D":
                f2 = smiles2rdkit2d 
            elif dst == "Morgan":
                f2 = smiles2morgan 
            elif dst == 'PubChem':
                f2 = smiles2pubchem
            self.func = lambda x:f2(f1(x)) 
        elif src in threeD_format:
            pass 


        ### load from xyz file, input is a filename (str), only contain one smiles 
        if src == 'XYZ' and dst == 'SMILES':
          self.func = xyzfile2smiles
        elif src == 'XYZ' and dst == 'SELFIES':
          self.func = xyzfile2selfies 
        elif src == 'XYZ' and dst == 'Graph3D':
          self.func = xyzfile2graph3d 
        elif src == 'XYZ' and dst == 'Coulumb':
          self.func = xyzfile2coulomb 

        ### SDF file 
        elif src == 'SDF' and dst == 'Graph3D':
          self.func = sdffile2graph3d_lst 
        elif src == 'SDF' and dst == 'SMILES':
          self.func = sdffile2smiles_lst  
        elif src == 'SDF' and dst == 'SELFIES':
          self.func = sdffile2selfies_lst 
        elif src == 'SDF' and dst == 'Coulumb':
          self.func = sdffile2coulomb



    def __call__(self, x):
      if type(x) == np.ndarray:
        x = x.tolist()

      if type(x) == str:
        if self.func != smiles2morgan:
          return self.func(x)
        else:
          return self.func(x, radius = self._radius, nBits = self._nbits)
      elif type(x) == list:
        if self.func != smiles2morgan:
          out = list(map(self.func, x))
        else:
          lst = []
          for x0 in x:
            lst.append(self.func(x0, radius = self._radius, nBits = self._nbits))
          out = lst 
        if self._dst in fingerprints_list:
          out = np.array(out)
        return out


    @staticmethod
    def eligible_format(src = None):
        '''
        given a src format, output all the available format of the src format
        Example
        MoleculeLink.eligible_format('SMILES')
        ## ['Graph', 'SMARTS', ...] 
        '''
        if src is not None:
          try:
            assert src in convert_dict
          except:
            raise Exception("src format is not supported")
          return convert_dict[src] 
        else:
          return convert_dict




######## test the MolConvert
# benzene = "c1ccccc1"
# convert = MolConvert(src = 'SMILES', dst = 'SELFIES')
# print(convert(benzene))


######## test the MoleculeFingerprint
# fps = ['ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem']
# smiles_lst = ['O=O', 'C', 'C#N', 'CC(=O)OC1=CC=CC=C1C(=O)O']
# for fp in fps:
#   MolFp = MoleculeFingerprint(fp = fp)
#   arr = MolFp(smiles_lst)
#   print(arr.shape, np.sum(arr))
#   arr = MolFp(smiles_lst[0])
#   print(arr.shape, np.sum(arr))
######## test the MoleculeFingerprint

class MolFilter:
  # MIT License: Checkout https://github.com/PatWalters/rd_filters
  def __init__(self, filters = 'all', property_filters_flag = True, HBA = [0, 10], HBD = [0, 5], LogP = [-5, 5], MW = [0, 500], Rot = [0, 10], TPSA = [0, 200]):
    try:
        from rd_filters.rd_filters import RDFilters, read_rules
    except:
        install('git+https://github.com/PatWalters/rd_filters.git')
        from rd_filters.rd_filters import RDFilters, read_rules
        
    import pkg_resources
    self.property_filters_flag = property_filters_flag
    all_filters = ['BMS', 'Dundee', 'Glaxo', 'Inpharmatica', 'LINT', 'MLSMR', 'PAINS', 'SureChEMBL']
    if filters == 'all':
      filters = all_filters
    else:
      if isinstance(filters, str):
        filters = [filters]
      if isinstance(filters, list):
        ## a set of filters
        for i in filters:
          if i not in all_filters:
            raise ValueError(i + " not found; Please choose from a list of available filters from 'BMS', 'Dundee', 'Glaxo', 'Inpharmatica', 'LINT', 'MLSMR', 'PAINS', 'SureChEMBL'")

    alert_file_name = pkg_resources.resource_filename('rd_filters', "data/alert_collection.csv")
    rules_file_path = pkg_resources.resource_filename('rd_filters', "data/rules.json")
    self.rf = RDFilters(alert_file_name)
    self.rule_dict = read_rules(rules_file_path)
    self.rule_dict['Rule_Inpharmatica'] = False
    for i in filters:
      self.rule_dict['Rule_'+ i] = True
    
    if self.property_filters_flag:
        self.rule_dict['HBA'], self.rule_dict['HBD'], self.rule_dict['LogP'], self.rule_dict['MW'], self.rule_dict['Rot'], self.rule_dict['TPSA'] = HBA, HBD, LogP, MW, Rot, TPSA
    else:
        del self.rule_dict['HBA'], self.rule_dict['HBD'], self.rule_dict['LogP'], self.rule_dict['MW'], self.rule_dict['Rot'], self.rule_dict['TPSA']
    print_sys("MolFilter is using the following filters:")

    for i,j in self.rule_dict.items():
      if i[:4] == 'Rule':
        if j:
          print_sys(i + ': ' + str(j))
      else:
        print_sys(i + ': ' + str(j))
    rule_list = [x.replace("Rule_", "") for x in self.rule_dict.keys() if x.startswith("Rule") and self.rule_dict[x]]
    rule_str = " and ".join(rule_list)
    self.rf.build_rule_list(rule_list)

  def __call__(self, input_data):
    import multiprocessing as mp
    from multiprocessing import Pool
    import pandas as pd

    if isinstance(input_data, str):
      input_data = [input_data]
    elif not isinstance(input_data, (list, np.ndarray, np.generic)):
      raise ValueError('Input must be a list/numpy array of SMILES or one SMILES string!')

    input_data = list(tuple(zip(input_data, list(range(len(input_data))))))

    num_cores = int(mp.cpu_count())
    p = Pool(num_cores)

    res = list(p.map(self.rf.evaluate, input_data))
    
    if self.property_filters_flag:
    
        df = pd.DataFrame(res, columns=["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"])
        df_ok = df[
            (df.FILTER == "OK") &
            df.MW.between(*self.rule_dict["MW"]) &
            df.LogP.between(*self.rule_dict["LogP"]) &
            df.HBD.between(*self.rule_dict["HBD"]) &
            df.HBA.between(*self.rule_dict["HBA"]) &
            df.TPSA.between(*self.rule_dict["TPSA"]) &
            df.Rot.between(*self.rule_dict["Rot"])
            ]
        
    else:
        df = pd.DataFrame(res, columns=["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"])
        df_ok = df[
            (df.FILTER == "OK")
            ]
    return df_ok.SMILES.values



