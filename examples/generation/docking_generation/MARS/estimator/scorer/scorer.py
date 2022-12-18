# modifed from: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py

import math
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
import networkx as nx

from ...common.chem import standardize_smiles
from . import sa_scorer, kinase_scorer  # , drd2_scorer, chemprop_scorer

import pyscreener
from tdc import Oracle

# oracle2 = Oracle(name = 'Docking_Score', software='vina', pyscreener_path = './', pdbids=['5WIU'], center=(-18.2, 14.4, -16.1), size=(15.4, 13.9, 14.5), buffer=10, path='./', num_worker=1, ncpu=4)
# oracle2 = Oracle(name = 'Docking_Score', software='vina', pyscreener_path = './', pdbids=['DRD3'], center=(-18.2, 14.4, -16.1), size=(15.4, 13.9, 14.5), buffer=10, path='./', num_worker=1, ncpu=4)

oracle2 = Oracle(
    name="Docking_Score",
    software="vina",
    pyscreener_path="./",
    receptors=["/project/molecular_data/graphnn/pyscreener/testing_inputs/DRD3.pdb"],
    center=(9, 22.5, 26),
    size=(15, 15, 15),
    buffer=10,
    path="./",
    num_worker=3,
    ncpu=8,
)


import random

### get scores
def get_scores(objective, mols, smiles2score=None):
    mols = [standardize_smiles(mol) for mol in mols]
    mols_valid = [mol for mol in mols if mol is not None]
    # print('get_scores')
    if objective == "drd2":
        scores = drd2_scorer.get_scores(mols_valid)
    elif objective == "jnk3" or objective == "gsk3b":
        scores = kinase_scorer.get_scores(objective, mols_valid)
    elif objective.startswith("chemprop"):
        scores = chemprop_scorer.get_scores(objective, mols_valid)
    else:
        scores = [get_score(objective, mol, smiles2score) for mol in mols_valid]

    scores = [scores.pop(0) if mol is not None else 0.0 for mol in mols]
    return scores


def get_score(objective, mol, smiles2score=None):
    try:
        if objective == "qed":
            # print('qed call')
            return QED.qed(mol)
        elif objective == "docking":
            smiles = Chem.MolToSmiles(mol)
            if smiles in smiles2score:
                return smiles2score[smiles]
            value = -oracle2(smiles)
            smiles2score[smiles] = value
            return value
            ####
        elif objective == "sa":
            # print('sa call')
            x = sa_scorer.calculateScore(mol)
            return (10.0 - x) / 9.0  # normalized to [0, 1]
        elif objective == "mw":  # molecular weight
            return mw(mol)
        elif objective == "logp":  # real number
            print("logp call")
            return Descriptors.MolLogP(mol)
        elif objective == "penalized_logp":
            print("plogp call")
            return penalized_logp(mol)
        elif "rand" in objective:
            raise NotImplementedError
            # return rand_scorer.get_score(objective, mol)
        else:
            raise NotImplementedError
    except ValueError:
        return 0.0


### molecular properties
def mw(mol):
    """
    molecular weight estimation from qed
    """
    x = Descriptors.MolWt(mol)
    a, b, c, d, e, f = 2.817, 392.575, 290.749, 2.420, 49.223, 65.371
    g = math.exp(-(x - c + d / 2) / e)
    h = math.exp(-(x - c - d / 2) / f)
    x = a + b / (1 + g) * (1 - 1 / (1 + h))
    return x / 104.981


def penalized_logp(mol):
    # Modified from https://github.com/bowenliu16/rl_graph_generation
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sa_scorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle
