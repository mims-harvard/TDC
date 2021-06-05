import dgl
import math
import copy
import rdkit
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from collections import defaultdict


### validity
def standardize_smiles(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None

def check_validity(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    if not isinstance(mol, Chem.Mol): return False
    if mol.GetNumBonds() < 1: return False
    try:
        # Chem.SanitizeMol(mol,
        #     sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        Chem.SanitizeMol(mol)
        Chem.RemoveHs(mol)
        return True
    except ValueError:
        return False


### breaking and combination
class Skeleton():
    def __init__(self, mol, u, bond_type=Chem.BondType.SINGLE):
        '''
        @params:
            mol       : submolecule of the arm, Chem.Mol
            u         : position to combine with arm, int
            bond_type : int
        '''
        self.mol = mol
        self.u = u
        self.bond_type = bond_type

class Arm():
    def __init__(self, mol, v, bond_type=Chem.BondType.SINGLE):
        '''
        @params:
            mol       : submolecule of the arm, Chem.Mol
            v         : position to combine with skeleton, int
            bond_type : int
        '''
        self.mol = mol
        self.v = v
        self.bond_type = bond_type

def break_bond(mol, u, v):
    '''
    break a bond in the molecule
    TODO: ValueError: Sanitization error: Can't kekulize mol.

    @params:
        mol : molecule of Chem.Mol
        u   : atom index on the skeleton
        v   : atom index on the arm
    @return:
        skeleton : molecule skeleton that contains u (w/o the virtual bond)
        arm      : molecule arm that contains v (with the virtual bond)
    '''
    mol = Chem.RWMol(copy.deepcopy(mol))
    bond = mol.GetBondBetweenAtoms(u, v)
    bond_type = bond.GetBondType()
    if not bond_type == \
        Chem.rdchem.BondType.SINGLE:
        raise ValueError
    mol.RemoveBond(u, v)

    mapping = []
    frags = list(Chem.rdmolops.GetMolFrags(mol,
        asMols=True, fragsMolAtomMapping=mapping))
    mapping = [list(m) for m in mapping]
    if not len(frags) == 2: raise ValueError
    if u not in mapping[0]:
        mapping = [mapping[1], mapping[0]]
        frags = [frags[1], frags[0]]

    # re-index
    u = mapping[0].index(u) 
    v = mapping[1].index(v)
    
    # standardizing frags will cause wrong indexing for u and v
    skeleton = Skeleton(frags[0], u, bond_type)
    arm = Arm(frags[1], v, bond_type)
    return skeleton, arm

def combine(skeleton, arm):
    '''
    combine a skeleton and an arm to form a complete molecule graph
    TODO: the smiles representations are different after combining,
          might because the node index is different
    TODO: unstandardized molecules fail to get features
    '''
    mol = Chem.CombineMols(skeleton.mol, arm.mol)
    mol = Chem.RWMol(mol)
    u = skeleton.u
    v = skeleton.mol.GetNumAtoms() + arm.v
    mol.AddBond(u, v, arm.bond_type)
    return mol.GetMol()


### data transformation
def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    nfp = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, nfp)
    return nfp

def mol_to_dgl(mol):
    '''
    @params:
        mol : Chem.Mol to transform
        plh : placeholder list to add arms
    '''
    # g = dgl.DGLGraph()
    g = dgl.graph([])

    # add nodes
    ATOM_TYPES = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    HYBRID_TYPES = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ]
    def zinc_nodes(mol):
        atom_feats_dict = defaultdict(list)
        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            atom = mol.GetAtomWithIdx(u)
            charge = atom.GetFormalCharge()
            symbol = atom.GetSymbol()
            atom_type = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            hybridization = atom.GetHybridization()
            num_h = atom.GetTotalNumHs()
            atom_feats_dict['node_type'].append(atom_type)
            atom_feats_dict['node_charge'].append(charge)

            h_u = []
            h_u += [
                int(symbol == x) 
                for x in ATOM_TYPES
            ]
            h_u.append(atom_type)
            h_u.append(int(charge))
            h_u.append(int(aromatic))
            h_u += [
                int(hybridization == x)
                for x in HYBRID_TYPES
            ]
            h_u.append(num_h)
            atom_feats_dict['n_feat'].append(torch.FloatTensor(h_u))

        atom_feats_dict['n_feat'] = torch.stack(atom_feats_dict['n_feat'], dim=0)
        atom_feats_dict['node_type'] = torch.LongTensor(atom_feats_dict['node_type'])
        atom_feats_dict['node_charge'] = torch.LongTensor(atom_feats_dict['node_charge'])
        return atom_feats_dict
    
    num_atoms = mol.GetNumAtoms()
    atom_feats = zinc_nodes(mol)
    g.add_nodes(num=num_atoms, data=atom_feats)

    # add edges, not complete
    BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC, None]
    def zinc_edges(mol, edges, self_loop=False):
        bond_feats_dict = defaultdict(list)
        edges = [idxs.tolist() for idxs in edges]
        for e in range(len(edges[0])):
            u, v = edges[0][e], edges[1][e]
            if u == v and not self_loop: continue

            e_uv = mol.GetBondBetweenAtoms(u, v)
            if e_uv is None: bond_type = None
            else: bond_type = e_uv.GetBondType()
            bond_feats_dict['e_feat'].append([
                float(bond_type == x)
                for x in BOND_TYPES
            ])

        bond_feats_dict['e_feat'] = torch.FloatTensor(
            bond_feats_dict['e_feat'])
        return bond_feats_dict
    
    bond_feats = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        g.add_edges([u, v], [v, u])
    bond_feats = zinc_edges(mol, g.edges())
    g.edata.update(bond_feats)
    return g
