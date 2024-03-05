# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import numpy as np
from typing import List

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.error")
    from rdkit.Chem.Fingerprints import FingerprintMols
    from rdkit.Chem import MACCSkeys
except:
    raise ImportError(
        "Please install rdkit by 'conda install -c conda-forge rdkit'! ")

from ...utils import print_sys
from ..oracle.oracle import (
    smiles_to_rdkit_mol,
    smiles_2_fingerprint_ECFP4,
    smiles_2_fingerprint_FCFP4,
    smiles_2_fingerprint_AP,
    smiles_2_fingerprint_ECFP6,
)
from ._smiles2pubchem import smiles2pubchem


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None


def smiles2morgan(s, radius=2, nBits=1024):
    """Convert smiles into Morgan Fingerprint.

    Args:
      smiles: str
      radius: int (default: 2)
      nBits: int (default: 1024)

    Returns:
      fp: numpy.array

    """
    try:
        s = canonicalize(s)
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol,
                                                             radius,
                                                             nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print_sys("rdkit not found this smiles for morgan: " + s +
                  " convert to all 0 features")
        features = np.zeros((nBits,))
    return features


def smiles2rdkit2d(s):
    """Convert smiles into 200-dim Normalized RDKit 2D vector.

    Args:
      smiles: str

    Returns:
      fp: numpy.array

    """

    s = canonicalize(s)
    try:
        from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
    except:
        raise ImportError(
            "Please install pip install git+https://github.com/bp-kelley/descriptastorus and pip install pandas-flavor"
        )
    try:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = np.array(generator.process(s)[1:])
        NaNs = np.isnan(features)
        features[NaNs] = 0
    except:
        print_sys("descriptastorus not found this smiles: " + s +
                  " convert to all 0 features")
        features = np.zeros((200,))
    return np.array(features)


def smiles2daylight(s):
    """Convert smiles into 2048-dim Daylight feature.

    Args:
      smiles: str

    Returns:
      fp: numpy.array

    """
    try:
        s = canonicalize(s)
        NumFinger = 2048
        mol = Chem.MolFromSmiles(s)
        bv = FingerprintMols.FingerprintMol(mol)
        temp = tuple(bv.GetOnBits())
        features = np.zeros((NumFinger,))
        features[np.array(temp)] = 1
    except:
        print_sys("rdkit not found this smiles: " + s +
                  " convert to all 0 features")
        features = np.zeros((2048,))
    return np.array(features)


def smiles2maccs(s):
    """Convert smiles into maccs feature.

    Args:
      smiles: str

    Returns:
      fp: numpy.array

    """
    s = canonicalize(s)
    mol = Chem.MolFromSmiles(s)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((0,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


"""
  ECFP2 ---- 1
  ECFP4 ---- 2
  ECFP6 ---- 3
  xxxxxxxxx  ------  https://github.com/rdkit/benchmarking_platform/blob/master/scoring/fingerprint_lib.py 

"""


def smiles2ECFP2(smiles):
    """Convert smiles into ECFP2 Morgan Fingerprint.

    Args:
      smiles: str

    Returns:
      fp: rdkit.DataStructs.cDataStructs.UIntSparseIntVect

    """
    nbits = 2048
    smiles = canonicalize(smiles)
    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 1, nBits=nbits)
    arr = np.zeros((0,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles2ECFP4(smiles):
    """Convert smiles into ECFP4 Morgan Fingerprint.

    Args:
      smiles: str

    Returns:
      fp: rdkit.DataStructs.cDataStructs.UIntSparseIntVect

    """
    nbits = 2048
    smiles = canonicalize(smiles)
    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=nbits)
    arr = np.zeros((0,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles2ECFP6(smiles):
    """Convert smiles into ECFP6 Morgan Fingerprint.

    Args:
      smiles: str, a SMILES string

    Returns:
      fp: rdkit.DataStructs.cDataStructs.UIntSparseIntVect

    refer: https://github.com/rdkit/benchmarking_platform/blob/master/scoring/fingerprint_lib.py

    """
    nbits = 2048
    smiles = canonicalize(smiles)
    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 3, nBits=nbits)
    arr = np.zeros((0,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# def smiles2smart(smiles):


class MoleculeFingerprint:
    """
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
    """

    def __init__(self, fp="ECFP4"):
        fp2func = {
            "ECFP2": smiles2ECFP2,
            "ECFP4": smiles2ECFP4,
            "ECFP6": smiles2ECFP6,
            "MACCS": smiles2maccs,
            "Daylight": smiles2daylight,
            "RDKit2D": smiles2rdkit2d,
            "Morgan": smiles2morgan,
            "PubChem": smiles2pubchem,
        }
        try:
            assert fp in fp2func
        except:
            raise Exception("The fingerprint you specify are not supported. \
              It can only among 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'"
                           )

        self.fp = fp
        self.func = fp2func[fp]

    def __call__(self, x):
        if type(x) == str:
            return self.func(x)
        elif type(x) == list:
            lst = list(map(self.func, x))
            arr = np.vstack(lst)
            return arr


def smiles2selfies(smiles):
    """Convert smiles into selfies.

    Args:
      smiles: str, a SMILES string

    Returns:
      selfies: str, a SELFIES string.

    """
    smiles = canonicalize(smiles)
    return sf.encoder(smiles)


def selfies2smiles(selfies):
    """Convert selfies into smiles.

    Args:
      selfies: str, a SELFIES string.

    Returns:
      smiles: str, a SMILES string

    """
    return canonicalize(sf.decoder(selfies))


def smiles2mol(smiles):
    """Convert SMILES string into rdkit.Chem.rdchem.Mol.

    Args:
      smiles: str, a SMILES string.

    Returns:
      mol: rdkit.Chem.rdchem.Mol

    """
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
    """convert SMILES string into two-dimensional molecular graph feature

    Args:
      smiles, str, a SMILES string

    Returns:
      idx2atom: dict, map from index to atom's symbol, e.g., {0:'C', 1:'N', ...}
      adj_matrix: np.array

    """
    smiles = canonicalize(smiles)
    mol = smiles2mol(smiles)
    n_atoms = mol.GetNumAtoms()
    idx2atom = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}
    adj_matrix = np.zeros((n_atoms, n_atoms), dtype=int)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        idx1 = a1.GetIdx()
        idx2 = a2.GetIdx()
        bond_type = bond.GetBondType()
        bond_idx = bondtype2idx(bond_type)
        adj_matrix[idx1, idx2] = bond_idx
        adj_matrix[idx2, idx1] = bond_idx
    return idx2atom, adj_matrix


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


############### PyG begin ###############
ELEM_LIST = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "Al",
    "I",
    "B",
    "K",
    "Se",
    "Zn",
    "H",
    "Cu",
    "Mn",
    "unknown",
]
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
# https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/chemutils.py


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom):
    return torch.Tensor(
        onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) +
        onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +
        onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3]) +
        [atom.GetIsAromatic()])


def smiles2PyG(smiles):
    """convert SMILES string into torch_geometric.data.Data

    Args:
      smiles, str, a SMILES string

    Returns:
      data, torch_geometric.data.Data

    """
    smiles = canonicalize(smiles)
    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    atom_features = torch.stack(atom_features)
    y = [atom.GetSymbol() for atom in mol.GetAtoms()]
    y = list(
        map(
            lambda x: ELEM_LIST.index(x)
            if x in ELEM_LIST else len(ELEM_LIST) - 1, y))
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


def molfile2PyG(molfile):
    smiles = molfile2smiles(molfile)
    smiles = canonicalize(smiles)
    return smiles2PyG(smiles)


############### PyG end ###############


############### DGL begin ###############
def smiles2DGL(smiles):
    """convert SMILES string into dgl.DGLGraph

    Args:
      smiles, str, a SMILES string

    Returns:
      g: dgl.DGLGraph()

    """
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


############### DGL end ###############

from ._xyz2mol import xyzfile2mol


def mol2smiles(mol):
    smiles = Chem.MolToSmiles(mol)
    smiles = canonicalize(smiles)
    return smiles


def xyzfile2smiles(xyzfile):
    """convert xyzfile into smiles string.

    Args:
      xyzfile: str, file

    Returns:
      smiles: str, a SMILES string

    """
    mol, _ = xyzfile2mol(xyzfile)
    smiles = mol2smiles(mol)
    smiles = canonicalize(smiles)
    return smiles


def xyzfile2selfies(xyzfile):
    """convert xyzfile into SELFIES string.

    Args:
      xyzfile: str, file

    Returns:
      selfies: str, a SELFIES string.

    """
    smiles = xyzfile2smiles(xyzfile)
    smiles = canonicalize(smiles)
    selfies = smiles2selfies(smiles)
    return selfies


def distance3d(coordinate_1, coordinate_2):
    return np.sqrt(
        sum([(c1 - c2)**2 for c1, c2 in zip(coordinate_1, coordinate_2)]))


def upper_atom(atomsymbol):
    return atomsymbol[0].upper() + atomsymbol[1:]


def xyzfile2graph3d(xyzfile):
    atoms, charge, xyz_coordinates = read_xyz_file(file)
    num_atoms = len(atoms)
    distance_adj_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = distance3d(xyz_coordinates[i], xyz_coordinates[j])
            distance_adj_matrix[i, j] = distance_adj_matrix[j, i] = distance
    idx2atom = {
        idx: upper_atom(str_atom(atom)) for idx, atom in enumerate(atoms)
    }
    mol, BO = xyzfile2mol(xyzfile)
    return idx2atom, distance_adj_matrix, BO


############## end xyz2mol ################


def sdffile2smiles_lst(sdffile):
    """convert SDF file into a list of SMILES string.

    Args:
      sdffile: str, file

    Returns:
      smiles_lst: a list of SMILES strings.

    """
    from rdkit.Chem.PandasTools import LoadSDF

    df = LoadSDF(sdffile, smilesName="SMILES")
    smiles_lst = df["SMILES"].to_list()
    return smiles_lst


def sdffile2mol_conformer(sdffile):
    """convert sdffile into a list of molecule conformers.

    Args:
      sdffile: str, file

    Returns:
      smiles_lst: a list of molecule conformers.

    """
    from rdkit.Chem.PandasTools import LoadSDF

    df = LoadSDF(sdffile, smilesName="SMILES")
    mol_lst = df["ROMol"].tolist()
    conformer_lst = []
    for mol in mol_lst:
        conformer = mol.GetConformer(id=0)
        conformer_lst.append(conformer)
    mol_conformer_lst = list(zip(mol_lst, conformer_lst))
    return mol_conformer_lst


def mol_conformer2graph3d(mol_conformer_lst):
    """convert list of (molecule, conformer) into a list of 3D graph.

    Args:
      mol_conformer_lst: list of tuple (molecule, conformer)

    Returns:
      graph3d_lst: a list of 3D graph.
            each graph has (i) idx2atom (dict); (ii) distance_adj_matrix (np.array); (iii) bondtype_adj_matrix (np.array)

    """
    graph3d_lst = []
    bond2num = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    for mol, conformer in mol_conformer_lst:
        atom_num = mol.GetNumAtoms()
        distance_adj_matrix = np.zeros((atom_num, atom_num))
        bondtype_adj_matrix = np.zeros((atom_num, atom_num), dtype=int)
        idx2atom = {i: v.GetSymbol() for i, v in enumerate(mol.GetAtoms())}
        positions = []
        for i in range(atom_num):
            pos = conformer.GetAtomPosition(i)
            coordinate = np.array([pos.x, pos.y, pos.z]).reshape(1, 3)
            positions.append(coordinate)
        positions = np.concatenate(positions, 0)
        for i in range(atom_num):
            for j in range(i + 1, atom_num):
                distance_adj_matrix[i,
                                    j] = distance_adj_matrix[j, i] = distance3d(
                                        positions[i], positions[j])
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            bondtype_adj_matrix[a1, a2] = bond2num[str(bt)]
            bondtype_adj_matrix[a1, a2] = bond2num[str(bt)]
        graph3d_lst.append((idx2atom, distance_adj_matrix, bondtype_adj_matrix))
    return graph3d_lst


def sdffile2graph3d_lst(sdffile):
    """convert SDF file into a list of 3D graph.

    Args:
      sdffile: SDF file

    Returns:
      graph3d_lst: a list of 3D graph.
            each graph has (i) idx2atom (dict); (ii) distance_adj_matrix (np.array); (iii) bondtype_adj_matrix (np.array)

    """
    mol_conformer_lst = sdffile2mol_conformer(sdffile)
    graph3d_lst = mol_conformer2graph3d(mol_conformer_lst)
    return graph3d_lst


def sdffile2selfies_lst(sdf):
    """convert sdffile into a list of SELFIES strings.

    Args:
      sdffile: str, file

    Returns:
      selfies_lst: a list of SELFIES strings.

    """
    smiles_lst = sdffile2smiles_lst(sdf)
    selfies_lst = list(map(smiles2selfies, smiles_lst))
    return selfies_lst


def smiles_lst2coulomb(smiles_lst):
    """convert a list of SMILES strings into coulomb format.

    Args:
      smiles_lst: a list of SELFIES strings.

    Returns:
      features: np.array

    """
    molecules = [Molecule(smiles, "smiles") for smiles in smiles_lst]
    for mol in molecules:
        mol.to_xyz(optimizer="UFF")
    cm = CoulombMatrix(cm_type="UM", n_jobs=-1)
    features = cm.represent(molecules)
    features = features.to_numpy()
    return features
    ## (nmol, max_atom_n**2),
    ## where max_atom_n is maximal number of atom in the smiles_lst
    ## features[i].reshape(max_atom_n, max_atom_n)[:3,:3]  -> 3*3 Coulomb matrix


def sdffile2coulomb(sdf):
    """convert sdffile into a list of coulomb feature.

    Args:
      sdffile: str, file

    Returns:
      coulomb feature: np.array
    """
    smiles_lst = sdffile2smiles_lst(sdf)
    return smiles_lst2coulomb(smiles_lst)


def xyzfile2coulomb(xyzfile):
    smiles = xyzfile2smiles(xyzfile)
    smiles = canonicalize(smiles)
    return smiles_lst2coulomb([smiles])


# 2D_format = ['SMILES', 'SELFIES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem']
# 3D_format = ['Graph3D', 'Coulumb']


## XXX2smiles
def molfile2smiles(molfile):
    """convert molfile into SMILES string

    Args:
      molfile: str, a file.

    Returns:
      smiles: str, SMILES strings

    """
    mol = Chem.MolFromMolFile(molfile)
    smiles = Chem.MolToSmiles(mol)
    smiles = canonicalize(smiles)
    return smiles


def mol2file2smiles(molfile):
    """convert mol2file into SMILES string

    Args:
      mol2file: str, a file.

    Returns:
      smiles: str, SMILES strings

    """
    mol = Chem.MolFromMol2File(molfile)
    smiles = Chem.MolToSmiles(mol)
    smiles = canonicalize(smiles)
    return smiles


## smiles2xxx

atom_types = ["C", "N", "O", "H", "F", "unknown"]  ### Cl, S?


def atom2onehot(atom):
    """convert atom to one-hot feature vector

    Args:
      'C'

    Returns:
      [1, 0, 0, 0, 0, ..]

    """
    onehot = np.zeros((1, len(atom_types)))
    idx = atom_types.index(atom)
    onehot[0, idx] = 1
    return onehot


def atomstring2atomfeature(atom_string_list):
    atom_features = [atom2onehot(atom) for atom in atom_string_list]
    atom_features = np.concatenate(atom_features, 0)
    return atom_features


def raw3D2pyg(raw3d_feature):
    """convert raw3d feature to pyg (torch-geometric) feature

    Args:
      raw3d_feature: (atom_string_list, positions, y)
        - atom_string_list: list, each element is an atom, length is N
        - positions: np.array, shape: (N,3)
        - y: float

    Returns:
      data = Data(x=x, pos=pos, y=y)

    """
    import torch
    from torch_geometric.data import Data  ### global

    # atom_string_list, positions, y = raw3d_feature
    atom_string_list, positions = raw3d_feature
    atom_features = atomstring2atomfeature(atom_string_list)
    atom_features = torch.from_numpy(atom_features)
    positions = torch.from_numpy(positions)
    # y = torch.FloatTensor(y)
    # data = Data(x = atom_features, pos = positions, y = y)
    data = Data(x=atom_features, pos=positions)
    return data


convert_dict = {
    "SMILES": [
        "SELFIES",
        "Graph2D",
        "PyG",
        "DGL",
        "ECFP2",
        "ECFP4",
        "ECFP6",
        "MACCS",
        "Daylight",
        "RDKit2D",
        "Morgan",
        "PubChem",
    ],
    "SELFIES": [
        "SMILES",
        "Graph2D",
        "PyG",
        "DGL",
        "ECFP2",
        "ECFP4",
        "ECFP6",
        "MACCS",
        "Daylight",
        "RDKit2D",
        "Morgan",
        "PubChem",
    ],
    "mol": [
        "SMILES",
        "SELFIES",
        "Graph2D",
        "PyG",
        "DGL",
        "ECFP2",
        "ECFP4",
        "ECFP6",
        "MACCS",
        "Daylight",
        "RDKit2D",
        "Morgan",
        "PubChem",
    ],
    "mol2": [
        "SMILES",
        "SELFIES",
        "Graph2D",
        "PyG",
        "DGL",
        "ECFP2",
        "ECFP4",
        "ECFP6",
        "MACCS",
        "Daylight",
        "RDKit2D",
        "Morgan",
        "PubChem",
    ],
    "SDF": ["SMILES", "SELFIES", "Graph3D", "Coulumb"],
    "XYZ": ["SMILES", "SELFIES", "Graph3D", "Coulumb"],
    "Raw3D": ["PyG3D"],
}

fingerprints_list = [
    "ECFP2",
    "ECFP4",
    "ECFP6",
    "MACCS",
    "Daylight",
    "RDKit2D",
    "Morgan",
    "PubChem",
]

twoD_format = [
    "SMILES",
    "SELFIES",
    "mol",
    "mol2",
]
threeD_format = [
    "SDF",
    "XYZ",
    "PyG3D",
    "Raw3D",
    "distance",
    "Coulumb",
    "shape",
]  ### shape:mesh


class MolConvert:
    """MolConvert: convert the molecule from src formet to dst format.




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
    """

    def __init__(self, src="SMILES", dst="Graph2D", radius=2, nBits=1024):
        self._src = src
        self._dst = dst
        self._radius = radius
        self._nbits = nBits

        self.convert_dict = convert_dict
        if "SELFIES" == src or "SELFIES" == dst:
            try:
                import selfies as sf

                global sf
            except:
                raise Exception(
                    "Please install selfies via 'pip install selfies'")

        if "Coulumb" == dst:
            try:
                from chemml.chem import CoulombMatrix, Molecule

                global CoulombMatrix, Molecule
            except:
                raise Exception(
                    "Please install chemml via 'pip install pybel' and 'pip install chemml'. "
                )

        if "PyG" == dst:
            try:
                import torch
                from torch_geometric.data import Data

                global torch
                global Data
            except:
                raise Exception(
                    "Please install PyTorch Geometric via 'https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html'."
                )

        if "DGL" == dst:
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
            raise Exception("It is not supported to convert src to dst.")

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
            if dst == "SMILES":
                f2 = canonicalize
            elif dst == "SELFIES":
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
            elif dst == "ECFP6":
                f2 = smiles2ECFP6
            elif dst == "MACCS":
                f2 = smiles2maccs
            elif dst == "Daylight":
                f2 = smiles2daylight
            elif dst == "RDKit2D":
                f2 = smiles2rdkit2d
            elif dst == "Morgan":
                f2 = smiles2morgan
            elif dst == "PubChem":
                f2 = smiles2pubchem
            self.func = lambda x: f2(f1(x))
        elif src in threeD_format:
            pass

        ### load from xyz file, input is a filename (str), only contain one smiles
        if src == "XYZ" and dst == "SMILES":
            self.func = xyzfile2smiles
        elif src == "XYZ" and dst == "SELFIES":
            self.func = xyzfile2selfies
        elif src == "XYZ" and dst == "Graph3D":
            self.func = xyzfile2graph3d
        elif src == "XYZ" and dst == "Coulumb":
            self.func = xyzfile2coulomb

        ### SDF file
        elif src == "SDF" and dst == "Graph3D":
            self.func = sdffile2graph3d_lst
        elif src == "SDF" and dst == "SMILES":
            self.func = sdffile2smiles_lst
        elif src == "SDF" and dst == "SELFIES":
            self.func = sdffile2selfies_lst
        elif src == "SDF" and dst == "Coulumb":
            self.func = sdffile2coulomb
        elif src == "Raw3D" and dst == "PyG3D":
            self.func = raw3D2pyg

    def __call__(self, x):
        if type(x) == np.ndarray:
            x = x.tolist()

        if type(x) == str:
            if self.func != smiles2morgan:
                return self.func(x)
            else:
                return self.func(x, radius=self._radius, nBits=self._nbits)
        elif type(x) == list:
            if self.func != smiles2morgan:
                out = list(map(self.func, x))
            else:
                lst = []
                for x0 in x:
                    lst.append(
                        self.func(x0, radius=self._radius, nBits=self._nbits))
                out = lst
            if self._dst in fingerprints_list:
                out = np.array(out)
            return out

    @staticmethod
    def eligible_format(src=None):
        """
        given a src format, output all the available format of the src format
        Example
        MoleculeLink.eligible_format('SMILES')
        ## ['Graph', 'SMARTS', ...]
        """
        if src is not None:
            try:
                assert src in convert_dict
            except:
                raise Exception("src format is not supported")
            return convert_dict[src]
        else:
            return convert_dict
