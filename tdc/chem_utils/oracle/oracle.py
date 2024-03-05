# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import pickle
import numpy as np
import os.path as op
from abc import abstractmethod
from functools import partial
from typing import List
import time, os, math, re
from packaging import version
import pkg_resources

try:
    import rdkit
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors
    import rdkit.Chem.QED as QED
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.error")
    from rdkit.Chem import rdMolDescriptors
    from rdkit.six import iteritems
except:
    raise ImportError(
        "Please install rdkit by 'conda install -c conda-forge rdkit'! ")

try:
    from scipy.stats.mstats import gmean
except:
    raise ImportError("Please install rdkit by 'pip install scipy'! ")

try:
    import networkx as nx
except:
    raise ImportError("Please install networkx by 'pip install networkx'! ")

from ...utils import oracle_load
from ...utils import print_sys, install

mean2func = {
    "geometric": gmean,
    "arithmetic": np.mean,
}
SKLEARN_VERSION = version.parse(
    pkg_resources.get_distribution("scikit-learn").version)


def smiles_to_rdkit_mol(smiles):
    """Convert smiles into rdkit's mol (molecule) format.

    Args:
      smiles: str, SMILES string.

    Returns:
      mol: rdkit.Chem.rdchem.Mol

    """
    mol = Chem.MolFromSmiles(smiles)
    #  Sanitization check (detects invalid valence)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
    return mol


def smiles_2_fingerprint_ECFP4(smiles):
    """Convert smiles into ECFP4 Morgan Fingerprint.

    Args:
      smiles: str, SMILES string.

    Returns:
      fp: rdkit.DataStructs.cDataStructs.UIntSparseIntVect

    """
    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetMorganFingerprint(molecule, 2)
    return fp


def smiles_2_fingerprint_FCFP4(smiles):
    """Convert smiles into FCFP4 Morgan Fingerprint.

    Args:
      smiles: str, SMILES string.

    Returns:
      fp: rdkit.DataStructs.cDataStructs.UIntSparseIntVect

    """
    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetMorganFingerprint(molecule, 2, useFeatures=True)
    return fp


def smiles_2_fingerprint_AP(smiles):
    """Convert smiles into Atom Pair Fingerprint.

    Args:
      smiles: str, SMILES string.

    Returns:
      fp: rdkit.DataStructs.cDataStructs.IntSparseIntVect

    """
    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetAtomPairFingerprint(molecule, maxLength=10)
    return fp


def smiles_2_fingerprint_ECFP6(smiles):
    """Convert smiles into ECFP6 Fingerprint.

    Args:
      smiles: str, SMILES string.

    Returns:
      fp: rdkit.DataStructs.cDataStructs.UIntSparseIntVect

    """
    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetMorganFingerprint(molecule, 3)
    return fp


fp2fpfunc = {
    "ECFP4": smiles_2_fingerprint_ECFP4,
    "FCFP4": smiles_2_fingerprint_FCFP4,
    "AP": smiles_2_fingerprint_AP,
    "ECFP6": smiles_2_fingerprint_ECFP6,
}


class ScoreModifier:
    """
    Interface for score modifiers.
    """

    @abstractmethod
    def __call__(self, x):
        """
        Apply the modifier on x.

        Args:
            x: float or np.array to modify

        Returns:
            float or np.array (depending on the type of x) after application of the distance function.
        """


class ChainedModifier(ScoreModifier):
    """
    Calls several modifiers one after the other, for instance:
        score = modifier3(modifier2(modifier1(raw_score)))
    """

    def __init__(self, modifiers: List[ScoreModifier]) -> None:
        """
        Args:
            modifiers: modifiers to call in sequence.
                The modifier applied last (and delivering the final score) is the last one in the list.
        """
        self.modifiers = modifiers

    def __call__(self, x):
        score = x
        for modifier in self.modifiers:
            score = modifier(score)
        return score


class LinearModifier(ScoreModifier):
    """
    Score modifier that multiplies the score by a scalar (default: 1, i.e. do nothing).
    """

    def __init__(self, slope=1.0):
        self.slope = slope

    def __call__(self, x):
        return self.slope * x


class SquaredModifier(ScoreModifier):
    """
    Score modifier that has a maximum at a given target value, and decreases
    quadratically with increasing distance from the target value.
    """

    def __init__(self, target_value: float, coefficient=1.0) -> None:
        self.target_value = target_value
        self.coefficient = coefficient

    def __call__(self, x):
        return 1.0 - self.coefficient * np.square(self.target_value - x)


class AbsoluteScoreModifier(ScoreModifier):
    """
    Score modifier that has a maximum at a given target value, and decreases
    linearly with increasing distance from the target value.
    """

    def __init__(self, target_value: float) -> None:
        self.target_value = target_value

    def __call__(self, x):
        return 1.0 - np.abs(self.target_value - x)


class GaussianModifier(ScoreModifier):
    """
    Score modifier that reproduces a Gaussian bell shape.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2.0))


class MinMaxGaussianModifier(ScoreModifier):
    """
    Score modifier that reproduces a half Gaussian bell shape.
    For minimize==True, the function is 1.0 for x <= mu and decreases to zero for x > mu.
    For minimize==False, the function is 1.0 for x >= mu and decreases to zero for x < mu.
    """

    def __init__(self, mu: float, sigma: float, minimize=False) -> None:
        self.mu = mu
        self.sigma = sigma
        self.minimize = minimize
        self._full_gaussian = GaussianModifier(mu=mu, sigma=sigma)

    def __call__(self, x):
        if self.minimize:
            mod_x = np.maximum(x, self.mu)
        else:
            mod_x = np.minimum(x, self.mu)
        return self._full_gaussian(mod_x)


MinGaussianModifier = partial(MinMaxGaussianModifier, minimize=True)
MaxGaussianModifier = partial(MinMaxGaussianModifier, minimize=False)


class ClippedScoreModifier(ScoreModifier):
    r"""
    Clips a score between specified low and high scores, and does a linear interpolation in between.

    This class works as follows:
    First the input is mapped onto a linear interpolation between both specified points.
    Then the generated values are clipped between low and high scores.
    """

    def __init__(self,
                 upper_x: float,
                 lower_x=0.0,
                 high_score=1.0,
                 low_score=0.0) -> None:
        """
        Args:
            upper_x: x-value from which (or until which if smaller than lower_x) the score is maximal
            lower_x: x-value until which (or from which if larger than upper_x) the score is minimal
            high_score: maximal score to clip to
            low_score: minimal score to clip to
        """
        assert low_score < high_score

        self.upper_x = upper_x
        self.lower_x = lower_x
        self.high_score = high_score
        self.low_score = low_score

        self.slope = (high_score - low_score) / (upper_x - lower_x)
        self.intercept = high_score - self.slope * upper_x

    def __call__(self, x):
        y = self.slope * x + self.intercept
        return np.clip(y, self.low_score, self.high_score)


class SmoothClippedScoreModifier(ScoreModifier):
    """
    Smooth variant of ClippedScoreModifier.

    Implemented as a logistic function that has the same steepness as ClippedScoreModifier in the
    center of the logistic function.
    """

    def __init__(self,
                 upper_x: float,
                 lower_x=0.0,
                 high_score=1.0,
                 low_score=0.0) -> None:
        """
        Args:
            upper_x: x-value from which (or until which if smaller than lower_x) the score approaches high_score
            lower_x: x-value until which (or from which if larger than upper_x) the score approaches low_score
            high_score: maximal score (reached at +/- infinity)
            low_score: minimal score (reached at -/+ infinity)
        """
        assert low_score < high_score

        self.upper_x = upper_x
        self.lower_x = lower_x
        self.high_score = high_score
        self.low_score = low_score

        # Slope of a standard logistic function in the middle is 0.25 -> rescale k accordingly
        self.k = 4.0 / (upper_x - lower_x)
        self.middle_x = (upper_x + lower_x) / 2
        self.L = high_score - low_score

    def __call__(self, x):
        return self.low_score + self.L / (1 + np.exp(-self.k *
                                                     (x - self.middle_x)))


class ThresholdedLinearModifier(ScoreModifier):
    """
    Returns a value of min(input, threshold)/threshold.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, x):
        return np.minimum(x, self.threshold) / self.threshold


# check the license for the code from readFragmentScores to CalculateScore here: https://github.com/EricTing/SAscore/blob/89d7689a85efed3cc918fb8ba6fe5cedf60b4a5a/src/sascorer.py#L134
_fscores = None


def readFragmentScores(name="fpscores"):
    import gzip

    global _fscores
    # generate the full path filename:
    # if name == "fpscores":
    #     name = op.join(previous_directory(op.dirname(__file__)), name)
    name = oracle_load("fpscores")
    try:
        with open("oracle/fpscores.pkl", "rb") as f:
            _fscores = pickle.load(f)
    except EOFError:
        import sys

        sys.exit(
            "TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/."
        )

    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(
        m, 2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (0.0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty -
              macrocyclePenalty)

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
    # smooth the 10-end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore


"""Scores based on an ECFP classifier for activity."""


def load_pickled_model(name: str):
    """
    Loading a pretrained model serialized with pickle.
    Usually for sklearn models.

    Args:
      name: Name of the model to load.

    Returns:
      The model.
    """

    try:
        with open(name, "rb") as f:
            model = pickle.load(f)
    except EOFError:
        import sys

        sys.exit(
            "TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/."
        )
    return model


# clf_model = None
def load_drd2_model():
    name = "oracle/drd2.pkl"

    if SKLEARN_VERSION >= version.parse("0.24.0"):
        name = "oracle/drd2_current.pkl"
    else:
        name = "oracle/drd2.pkl"

    return load_pickled_model(name)


def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx, v in fp.GetNonzeroElements().items():
        nidx = idx % size
        nfp[0, nidx] += int(v)
    return nfp


def drd2(smile):
    """Evaluate DRD2 score of a SMILES string

    Args:
      smiles: str

    Returns:
      drd_score: float

    """

    if "drd2_model" not in globals().keys():
        global drd2_model
        drd2_model = load_drd2_model()

    mol = Chem.MolFromSmiles(smile)
    if mol:
        fp = fingerprints_from_mol(mol)
        score = drd2_model.predict_proba(fp)[:, 1]
        drd_score = float(score)
        return drd_score
    return 0.0


def load_cyp3a4_veith():
    oracle_file = "oracle/cyp3a4_veith.pkl"
    return load_pickled_model(oracle_file)


def cyp3a4_veith(smiles):
    try:
        from DeepPurpose import utils
    except:
        raise ImportError(
            "Please install DeepPurpose by 'pip install DeepPurpose'")

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if "cyp3a4_veith_model" not in globals().keys():
        global cyp3a4_veith_model
        cyp3a4_veith_model = load_cyp3a4_veith()

    import warnings, os

    warnings.filterwarnings("ignore")

    X_drug = [smiles]
    drug_encoding = "CNN"
    y = [1]
    X_pred = utils.data_process(X_drug=X_drug,
                                y=y,
                                drug_encoding=drug_encoding,
                                split_method="no_split")
    # cyp3a4_veith_model = cyp3a4_veith_model.to("cuda:0")
    y_pred = cyp3a4_veith_model.predict(X_pred)
    return y_pred[0]


## from https://github.com/wengong-jin/iclr19-graph2graph/blob/master/props/properties.py
## from https://github.com/wengong-jin/multiobj-rationale/blob/master/properties.py


def similarity(smiles_a, smiles_b):
    """Evaluate Tanimoto similarity between 2 SMILES strings

    Args:
      smiles_a: str, SMILES string
      smiles_b: str, SMILES string

    Returns:
      similarity score: float, between 0 and 1.

    """
    if smiles_a is None or smiles_b is None:
        return 0.0
    amol = Chem.MolFromSmiles(smiles_a)
    bmol = Chem.MolFromSmiles(smiles_b)
    if amol is None or bmol is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol,
                                                2,
                                                nBits=2048,
                                                useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol,
                                                2,
                                                nBits=2048,
                                                useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def qed(smiles):
    """Evaluate QED score of a SMILES string

    Args:
      smiles: str

    Returns:
      qed_score: float, between 0 and 1.

    """
    if smiles is None:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return QED.qed(mol)


def penalized_logp(s):
    """Evaluate LogP score of a SMILES string

    Args:
      smiles: str

    Returns:
      logp_score: float, between - infinity and + infinity

    """
    if s is None:
        return -100.0
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455
    log_p = Descriptors.MolLogP(mol)
    # SA = -sascorer.calculateScore(mol)
    SA = -calculateScore(mol)

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


def SA(s):
    """Evaluate SA score of a SMILES string

    Args:
      smiles: str

    Returns:
      SAscore: float

    """
    if s is None:
        return 100
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return 100
    SAscore = calculateScore(mol)
    return SAscore


def load_gsk3b_model():
    gsk3_model_path = "oracle/gsk3b.pkl"
    if SKLEARN_VERSION >= version.parse("0.24.0"):
        gsk3_model_path = "oracle/gsk3b_current.pkl"
    return load_pickled_model(gsk3_model_path)


def gsk3b(smiles):
    """Evaluate GSK3B score of a SMILES string

    Args:
      smiles: str

    Returns:
      gsk3_score: float, between 0 and 1.

    """
    if "gsk3_model" not in globals().keys():
        global gsk3_model
        gsk3_model = load_gsk3b_model()

    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, features)
    fp = features.reshape(1, -1)
    gsk3_score = gsk3_model.predict_proba(fp)[0, 1]
    return gsk3_score


class jnk3:
    """Evaluate JSK3 score of a SMILES string

    Args:
      smiles: str

    Returns:
      jnk3_score: float , between 0 and 1.

    """

    def __init__(self):

        jnk3_model_path = "oracle/jnk3.pkl"
        if SKLEARN_VERSION >= version.parse("0.24.0"):
            jnk3_model_path = "oracle/jnk3_current.pkl"
        self.jnk3_model = load_pickled_model(jnk3_model_path)

    def __call__(self, smiles):
        molecule = smiles_to_rdkit_mol(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, features)
        fp = features.reshape(1, -1)
        jnk3_score = self.jnk3_model.predict_proba(fp)[0, 1]
        return jnk3_score


class AtomCounter:

    def __init__(self, element):
        """
        Args:
            element: element to count within a molecule
        """
        self.element = element

    def __call__(self, mol):
        """
        Count the number of atoms of a given type.

        Args:
            mol: molecule

        Returns:
            The number of atoms of the given type.
        """
        # if the molecule contains H atoms, they may be implicit, so add them
        if self.element == "H":
            mol = Chem.AddHs(mol)

        return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == self.element)


def parse_molecular_formula(formula):
    """
    Parse a molecular formulat to get the element types and counts.

    Args:
        formula: molecular formula, f.i. "C8H3F3Br"

    Returns:
        A list of tuples containing element types and number of occurrences.
    """
    import re

    matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)

    # Convert matches to the required format
    results = []
    for match in matches:
        # convert count to an integer, and set it to 1 if the count is not visible in the molecular formula
        count = 1 if not match[1] else int(match[1])
        results.append((match[0], count))

    return results


def smiles2formula(smiles):

    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    mol = Chem.MolFromSmiles(smiles)
    formula = CalcMolFormula(mol)
    return formula


def canonicalize(smiles: str, include_stereocenters=True):
    """
    Canonicalize the SMILES strings with RDKit.

    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543

    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None


class Isomer_scoring_prev:

    def __init__(self, target_smiles, means="geometric"):
        assert means in ["geometric", "arithmetic"]
        if means == "geometric":
            self.mean_func = gmean
        else:
            self.mean_func = np.mean
        atom2cnt_lst = parse_molecular_formula(target_smiles)
        total_atom_num = sum([cnt for atom, cnt in atom2cnt_lst])
        self.total_atom_modifier = GaussianModifier(mu=total_atom_num,
                                                    sigma=2.0)
        self.AtomCounter_Modifier_lst = [((AtomCounter(atom)),
                                          GaussianModifier(mu=cnt, sigma=1.0))
                                         for atom, cnt in atom2cnt_lst]

    def __call__(self, test_smiles):
        molecule = smiles_to_rdkit_mol(test_smiles)
        all_scores = []
        for atom_counter, modifier_func in self.AtomCounter_Modifier_lst:
            all_scores.append(modifier_func(atom_counter(molecule)))

        ### total atom number
        atom2cnt_lst = parse_molecular_formula(test_smiles)
        # ## todo add Hs
        total_atom_num = sum([cnt for atom, cnt in atom2cnt_lst])
        all_scores.append(self.total_atom_modifier(total_atom_num))
        return self.mean_func(all_scores)


class Isomer_scoring:

    def __init__(self, target_smiles, means="geometric"):
        assert means in ["geometric", "arithmetic"]
        if means == "geometric":
            self.mean_func = gmean
        else:
            self.mean_func = np.mean
        atom2cnt_lst = parse_molecular_formula(target_smiles)
        total_atom_num = sum([cnt for atom, cnt in atom2cnt_lst])
        self.total_atom_modifier = GaussianModifier(mu=total_atom_num,
                                                    sigma=2.0)
        self.AtomCounter_Modifier_lst = [((AtomCounter(atom)),
                                          GaussianModifier(mu=cnt, sigma=1.0))
                                         for atom, cnt in atom2cnt_lst]

    def __call__(self, test_smiles):
        #### difference 1
        #### add hydrogen atoms
        test_smiles = canonicalize(test_smiles)
        test_mol = Chem.MolFromSmiles(test_smiles)
        test_mol2 = Chem.AddHs(test_mol)
        test_smiles = Chem.MolToSmiles(test_mol2)

        molecule = smiles_to_rdkit_mol(test_smiles)
        all_scores = []
        for atom_counter, modifier_func in self.AtomCounter_Modifier_lst:
            all_scores.append(modifier_func(atom_counter(molecule)))

        #### difference 2
        ### total atom number
        test_formula = smiles2formula(test_smiles)
        atom2cnt_lst = parse_molecular_formula(test_formula)
        # atom2cnt_lst = parse_molecular_formula(test_smiles)
        # ## todo add Hs
        total_atom_num = sum([cnt for atom, cnt in atom2cnt_lst])
        all_scores.append(self.total_atom_modifier(total_atom_num))
        return self.mean_func(all_scores)


def isomer_meta_prev(target_smiles, means="geometric"):
    return Isomer_scoring_prev(target_smiles, means=means)


def isomer_meta(target_smiles, means="geometric"):
    return Isomer_scoring(target_smiles, means=means)


isomers_c7h8n2o2_prev = isomer_meta_prev(target_smiles="C7H8N2O2",
                                         means="geometric")
isomers_c9h10n2o2pf2cl_prev = isomer_meta_prev(target_smiles="C9H10N2O2PF2Cl",
                                               means="geometric")
isomers_c11h24_prev = isomer_meta_prev(target_smiles="C11H24",
                                       means="geometric")

isomers_c7h8n2o2 = isomer_meta(target_smiles="C7H8N2O2", means="geometric")
isomers_c9h10n2o2pf2cl = isomer_meta(target_smiles="C9H10N2O2PF2Cl",
                                     means="geometric")
isomers_c11h24 = isomer_meta(target_smiles="C11H24", means="geometric")


class rediscovery_meta:

    def __init__(self, target_smiles, fp="ECFP4"):
        self.similarity_func = fp2fpfunc[fp]
        self.target_fp = self.similarity_func(target_smiles)

    def __call__(self, test_smiles):
        test_fp = self.similarity_func(test_smiles)
        similarity_value = DataStructs.TanimotoSimilarity(
            self.target_fp, test_fp)
        return similarity_value


class similarity_meta:

    def __init__(self, target_smiles, fp="FCFP4", modifier_func=None):
        self.similarity_func = fp2fpfunc[fp]
        self.target_fp = self.similarity_func(target_smiles)
        self.modifier_func = modifier_func

    def __call__(self, test_smiles):
        test_fp = self.similarity_func(test_smiles)
        similarity_value = DataStructs.TanimotoSimilarity(
            self.target_fp, test_fp)
        if self.modifier_func is None:
            modifier_score = similarity_value
        else:
            modifier_score = self.modifier_func(similarity_value)
        return modifier_score


celecoxib_rediscovery = rediscovery_meta(
    target_smiles="CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
    fp="ECFP4")
troglitazone_rediscovery = rediscovery_meta(
    target_smiles="Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
    fp="ECFP4")
thiothixene_rediscovery = rediscovery_meta(
    target_smiles="CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",
    fp="ECFP4")

similarity_modifier = ClippedScoreModifier(upper_x=0.75)
aripiprazole_similarity = similarity_meta(
    target_smiles="Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl",
    fp="FCFP4",
    modifier_func=similarity_modifier,
)

albuterol_similarity = similarity_meta(
    target_smiles="CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    fp="FCFP4",
    modifier_func=similarity_modifier,
)

mestranol_similarity = similarity_meta(
    target_smiles="COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",
    fp="AP",
    modifier_func=similarity_modifier,
)


class median_meta:

    def __init__(
        self,
        target_smiles_1,
        target_smiles_2,
        fp1="ECFP6",
        fp2="ECFP6",
        modifier_func1=None,
        modifier_func2=None,
        means="geometric",
    ):
        self.similarity_func1 = fp2fpfunc[fp1]
        self.similarity_func2 = fp2fpfunc[fp2]
        self.target_fp1 = self.similarity_func1(target_smiles_1)
        self.target_fp2 = self.similarity_func2(target_smiles_2)
        self.modifier_func1 = modifier_func1
        self.modifier_func2 = modifier_func2
        assert means in ["geometric", "arithmetic"]
        self.mean_func = mean2func[means]

    def __call__(self, test_smiles):
        test_fp1 = self.similarity_func1(test_smiles)
        test_fp2 = (test_fp1 if self.similarity_func2 == self.similarity_func1
                    else self.similarity_func2(test_smiles))
        similarity_value1 = DataStructs.TanimotoSimilarity(
            self.target_fp1, test_fp1)
        similarity_value2 = DataStructs.TanimotoSimilarity(
            self.target_fp2, test_fp2)
        if self.modifier_func1 is None:
            modifier_score1 = similarity_value1
        else:
            modifier_score1 = self.modifier_func1(similarity_value1)
        if self.modifier_func2 is None:
            modifier_score2 = similarity_value2
        else:
            modifier_score2 = self.modifier_func2(similarity_value2)
        final_score = self.mean_func([modifier_score1, modifier_score2])
        return final_score


camphor_smiles = "CC1(C)C2CCC1(C)C(=O)C2"
menthol_smiles = "CC(C)C1CCC(C)CC1O"

median1 = median_meta(
    target_smiles_1=camphor_smiles,
    target_smiles_2=menthol_smiles,
    fp1="ECFP4",
    fp2="ECFP4",
    modifier_func1=None,
    modifier_func2=None,
    means="geometric",
)

tadalafil_smiles = "O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C"
sildenafil_smiles = "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"
median2 = median_meta(
    target_smiles_1=tadalafil_smiles,
    target_smiles_2=sildenafil_smiles,
    fp1="ECFP6",
    fp2="ECFP6",
    modifier_func1=None,
    modifier_func2=None,
    means="geometric",
)


class MPO_meta:

    def __init__(self, means):
        """
        target_smiles, fp in ['ECFP4', 'AP', ..., ]
        scoring,
        modifier,

        """

        assert means in ["geometric", "arithmetic"]
        self.mean_func = mean2func[means]

    def __call__(self, test_smiles):
        molecule = smiles_to_rdkit_mol(test_smiles)

        score_lst = []
        return self.mean_func(score_lst)


def osimertinib_mpo(test_smiles):

    if "osimertinib_fp_fcfc4" not in globals().keys():
        global osimertinib_fp_fcfc4, osimertinib_fp_ecfc6
        osimertinib_smiles = (
            "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34")
        osimertinib_fp_fcfc4 = smiles_2_fingerprint_FCFP4(osimertinib_smiles)
        osimertinib_fp_ecfc6 = smiles_2_fingerprint_ECFP6(osimertinib_smiles)

    sim_v1_modifier = ClippedScoreModifier(upper_x=0.8)
    sim_v2_modifier = MinGaussianModifier(mu=0.85, sigma=0.1)
    tpsa_modifier = MaxGaussianModifier(mu=100, sigma=10)
    logp_modifier = MinGaussianModifier(mu=1, sigma=1)

    molecule = smiles_to_rdkit_mol(test_smiles)
    fp_fcfc4 = smiles_2_fingerprint_FCFP4(test_smiles)
    fp_ecfc6 = smiles_2_fingerprint_ECFP6(test_smiles)
    tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
    logp_score = logp_modifier(Descriptors.MolLogP(molecule))
    similarity_v1 = sim_v1_modifier(
        DataStructs.TanimotoSimilarity(osimertinib_fp_fcfc4, fp_fcfc4))
    similarity_v2 = sim_v2_modifier(
        DataStructs.TanimotoSimilarity(osimertinib_fp_ecfc6, fp_ecfc6))

    osimertinib_gmean = gmean(
        [tpsa_score, logp_score, similarity_v1, similarity_v2])
    return osimertinib_gmean


def fexofenadine_mpo(test_smiles):
    if "fexofenadine_fp" not in globals().keys():
        global fexofenadine_fp
        fexofenadine_smiles = (
            "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4")
        fexofenadine_fp = smiles_2_fingerprint_AP(fexofenadine_smiles)

    similar_modifier = ClippedScoreModifier(upper_x=0.8)
    tpsa_modifier = MaxGaussianModifier(mu=90, sigma=10)
    logp_modifier = MinGaussianModifier(mu=4, sigma=1)

    molecule = smiles_to_rdkit_mol(test_smiles)
    fp_ap = smiles_2_fingerprint_AP(test_smiles)
    tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
    logp_score = logp_modifier(Descriptors.MolLogP(molecule))
    similarity_value = similar_modifier(
        DataStructs.TanimotoSimilarity(fp_ap, fexofenadine_fp))
    fexofenadine_gmean = gmean([tpsa_score, logp_score, similarity_value])
    return fexofenadine_gmean


def ranolazine_mpo(test_smiles):
    if "ranolazine_fp" not in globals().keys():
        global ranolazine_fp, fluorine_counter
        ranolazine_smiles = "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2"
        ranolazine_fp = smiles_2_fingerprint_AP(ranolazine_smiles)
        fluorine_counter = AtomCounter("F")

    similar_modifier = ClippedScoreModifier(upper_x=0.7)
    tpsa_modifier = MaxGaussianModifier(mu=95, sigma=20)
    logp_modifier = MaxGaussianModifier(mu=7, sigma=1)
    fluorine_modifier = GaussianModifier(mu=1, sigma=1.0)

    molecule = smiles_to_rdkit_mol(test_smiles)
    fp_ap = smiles_2_fingerprint_AP(test_smiles)
    tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
    logp_score = logp_modifier(Descriptors.MolLogP(molecule))
    similarity_value = similar_modifier(
        DataStructs.TanimotoSimilarity(fp_ap, ranolazine_fp))
    fluorine_value = fluorine_modifier(fluorine_counter(molecule))

    ranolazine_gmean = gmean(
        [tpsa_score, logp_score, similarity_value, fluorine_value])
    return ranolazine_gmean


def perindopril_mpo(test_smiles):
    ## no similar_modifier

    if "perindopril_fp" not in globals().keys():
        global perindopril_fp, num_aromatic_rings
        perindopril_smiles = "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"
        perindopril_fp = smiles_2_fingerprint_ECFP4(perindopril_smiles)

        def num_aromatic_rings(mol):
            return rdMolDescriptors.CalcNumAromaticRings(mol)

    arom_rings_modifier = GaussianModifier(mu=2, sigma=0.5)

    molecule = smiles_to_rdkit_mol(test_smiles)
    fp_ecfp4 = smiles_2_fingerprint_ECFP4(test_smiles)

    similarity_value = DataStructs.TanimotoSimilarity(fp_ecfp4, perindopril_fp)
    num_aromatic_rings_value = arom_rings_modifier(num_aromatic_rings(molecule))

    perindopril_gmean = gmean([similarity_value, num_aromatic_rings_value])
    return perindopril_gmean


def amlodipine_mpo(test_smiles):
    ## no similar_modifier
    if "amlodipine_fp" not in globals().keys():
        global amlodipine_fp, num_rings
        amlodipine_smiles = "Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC"
        amlodipine_fp = smiles_2_fingerprint_ECFP4(amlodipine_smiles)

        def num_rings(mol):
            return rdMolDescriptors.CalcNumRings(mol)

    num_rings_modifier = GaussianModifier(mu=3, sigma=0.5)

    molecule = smiles_to_rdkit_mol(test_smiles)
    fp_ecfp4 = smiles_2_fingerprint_ECFP4(test_smiles)

    similarity_value = DataStructs.TanimotoSimilarity(fp_ecfp4, amlodipine_fp)
    num_rings_value = num_rings_modifier(num_rings(molecule))

    amlodipine_gmean = gmean([similarity_value, num_rings_value])
    return amlodipine_gmean


def zaleplon_mpo_prev(test_smiles):
    if "zaleplon_fp" not in globals().keys():
        global zaleplon_fp, isomer_scoring_C19H17N3O2
        zaleplon_smiles = "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1"
        zaleplon_fp = smiles_2_fingerprint_ECFP4(zaleplon_smiles)
        isomer_scoring_C19H17N3O2 = Isomer_scoring_prev(
            target_smiles="C19H17N3O2")

    fp = smiles_2_fingerprint_ECFP4(test_smiles)
    similarity_value = DataStructs.TanimotoSimilarity(fp, zaleplon_fp)
    isomer_value = isomer_scoring_C19H17N3O2(test_smiles)
    return gmean([similarity_value, isomer_value])


def zaleplon_mpo(test_smiles):
    if "zaleplon_fp" not in globals().keys():
        global zaleplon_fp, isomer_scoring_C19H17N3O2
        zaleplon_smiles = "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1"
        zaleplon_fp = smiles_2_fingerprint_ECFP4(zaleplon_smiles)
        isomer_scoring_C19H17N3O2 = Isomer_scoring(target_smiles="C19H17N3O2")

    fp = smiles_2_fingerprint_ECFP4(test_smiles)
    similarity_value = DataStructs.TanimotoSimilarity(fp, zaleplon_fp)
    isomer_value = isomer_scoring_C19H17N3O2(test_smiles)
    return gmean([similarity_value, isomer_value])


def sitagliptin_mpo_prev(test_smiles):
    if "sitagliptin_fp_ecfp4" not in globals().keys():
        global sitagliptin_fp_ecfp4, sitagliptin_logp_modifier, sitagliptin_tpsa_modifier, isomers_scoring_C16H15F6N5O, sitagliptin_similar_modifier
        sitagliptin_smiles = "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"
        sitagliptin_fp_ecfp4 = smiles_2_fingerprint_ECFP4(sitagliptin_smiles)
        sitagliptin_mol = Chem.MolFromSmiles(sitagliptin_smiles)
        sitagliptin_logp = Descriptors.MolLogP(sitagliptin_mol)
        sitagliptin_tpsa = Descriptors.TPSA(sitagliptin_mol)
        sitagliptin_logp_modifier = GaussianModifier(mu=sitagliptin_logp,
                                                     sigma=0.2)
        sitagliptin_tpsa_modifier = GaussianModifier(mu=sitagliptin_tpsa,
                                                     sigma=5)
        isomers_scoring_C16H15F6N5O = Isomer_scoring_prev("C16H15F6N5O")
        sitagliptin_similar_modifier = GaussianModifier(mu=0, sigma=0.1)

    molecule = Chem.MolFromSmiles(test_smiles)
    fp_ecfp4 = smiles_2_fingerprint_ECFP4(test_smiles)
    logp_score = Descriptors.MolLogP(molecule)
    logp_score = sitagliptin_logp_modifier(logp_score)
    tpsa_score = Descriptors.TPSA(molecule)
    tpsa_score = sitagliptin_tpsa_modifier(tpsa_score)
    isomer_score = isomers_scoring_C16H15F6N5O(test_smiles)
    similarity_value = sitagliptin_similar_modifier(
        DataStructs.TanimotoSimilarity(fp_ecfp4, sitagliptin_fp_ecfp4))
    return gmean([similarity_value, logp_score, tpsa_score, isomer_score])


def sitagliptin_mpo(test_smiles):
    if "sitagliptin_fp_ecfp4" not in globals().keys():
        global sitagliptin_fp_ecfp4, sitagliptin_logp_modifier, sitagliptin_tpsa_modifier, isomers_scoring_C16H15F6N5O, sitagliptin_similar_modifier
        sitagliptin_smiles = "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"
        sitagliptin_fp_ecfp4 = smiles_2_fingerprint_ECFP4(sitagliptin_smiles)
        sitagliptin_mol = Chem.MolFromSmiles(sitagliptin_smiles)
        sitagliptin_logp = Descriptors.MolLogP(sitagliptin_mol)
        sitagliptin_tpsa = Descriptors.TPSA(sitagliptin_mol)
        sitagliptin_logp_modifier = GaussianModifier(mu=sitagliptin_logp,
                                                     sigma=0.2)
        sitagliptin_tpsa_modifier = GaussianModifier(mu=sitagliptin_tpsa,
                                                     sigma=5)
        isomers_scoring_C16H15F6N5O = Isomer_scoring("C16H15F6N5O")
        sitagliptin_similar_modifier = GaussianModifier(mu=0, sigma=0.1)

    molecule = Chem.MolFromSmiles(test_smiles)
    fp_ecfp4 = smiles_2_fingerprint_ECFP4(test_smiles)
    logp_score = Descriptors.MolLogP(molecule)
    logp_score = sitagliptin_logp_modifier(logp_score)
    tpsa_score = Descriptors.TPSA(molecule)
    tpsa_score = sitagliptin_tpsa_modifier(tpsa_score)
    isomer_score = isomers_scoring_C16H15F6N5O(test_smiles)
    similarity_value = sitagliptin_similar_modifier(
        DataStructs.TanimotoSimilarity(fp_ecfp4, sitagliptin_fp_ecfp4))
    return gmean([similarity_value, logp_score, tpsa_score, isomer_score])


def get_PHCO_fingerprint(mol):
    if "Gobbi_Pharm2D" not in globals().keys():
        global Gobbi_Pharm2D, Generate
        from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
    return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)


class SMARTS_scoring:

    def __init__(self, target_smarts, inverse):
        self.target_mol = Chem.MolFromSmarts(target_smarts)
        self.inverse = inverse

    def __call__(self, mol):
        matches = mol.GetSubstructMatches(self.target_mol)
        if len(matches) > 0:
            if self.inverse:
                return 0.0
            else:
                return 1.0
        else:
            if self.inverse:
                return 1.0
            else:
                return 0.0


def deco_hop(test_smiles):
    if "pharmacophor_fp" not in globals().keys():
        global pharmacophor_fp, deco1_smarts_scoring, deco2_smarts_scoring, scaffold_smarts_scoring
        pharmacophor_smiles = "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"
        pharmacophor_mol = smiles_to_rdkit_mol(pharmacophor_smiles)
        pharmacophor_fp = get_PHCO_fingerprint(pharmacophor_mol)

        deco1_smarts_scoring = SMARTS_scoring(target_smarts="CS([#6])(=O)=O",
                                              inverse=True)
        deco2_smarts_scoring = SMARTS_scoring(
            target_smarts="[#7]-c1ccc2ncsc2c1", inverse=True)
        scaffold_smarts_scoring = SMARTS_scoring(
            target_smarts="[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12",
            inverse=False,
        )

    molecule = smiles_to_rdkit_mol(test_smiles)
    fp = get_PHCO_fingerprint(molecule)
    similarity_modifier = ClippedScoreModifier(upper_x=0.85)

    similarity_value = similarity_modifier(
        DataStructs.TanimotoSimilarity(fp, pharmacophor_fp))
    deco1_score = deco1_smarts_scoring(molecule)
    deco2_score = deco2_smarts_scoring(molecule)
    scaffold_score = scaffold_smarts_scoring(molecule)

    all_scores = np.mean(
        [similarity_value, deco1_score, deco2_score, scaffold_score])
    return all_scores


def scaffold_hop(test_smiles):
    if ("pharmacophor_fp" not in globals().keys() or
            "scaffold_smarts_scoring" not in globals().keys() or
            "deco_smarts_scoring" not in globals().keys()):
        global pharmacophor_fp, deco_smarts_scoring, scaffold_smarts_scoring
        pharmacophor_smiles = "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"
        pharmacophor_mol = smiles_to_rdkit_mol(pharmacophor_smiles)
        pharmacophor_fp = get_PHCO_fingerprint(pharmacophor_mol)

        deco_smarts_scoring = SMARTS_scoring(
            target_smarts=
            "[#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1",
            inverse=False,
        )

        scaffold_smarts_scoring = SMARTS_scoring(
            target_smarts="[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12",
            inverse=True)

    molecule = smiles_to_rdkit_mol(test_smiles)
    fp = get_PHCO_fingerprint(molecule)
    similarity_modifier = ClippedScoreModifier(upper_x=0.75)

    similarity_value = similarity_modifier(
        DataStructs.TanimotoSimilarity(fp, pharmacophor_fp))
    deco_score = deco_smarts_scoring(molecule)
    scaffold_score = scaffold_smarts_scoring(molecule)

    all_scores = np.mean([similarity_value, deco_score, scaffold_score])
    return all_scores


def valsartan_smarts(test_smiles):
    if "valsartan_logp_modifier" not in globals().keys():
        global valsartan_mol, valsartan_logp_modifier, valsartan_tpsa_modifier, valsartan_bertz_modifier
        valsartan_smarts = "CN(C=O)Cc1ccc(c2ccccc2)cc1"  ### smarts
        valsartan_mol = Chem.MolFromSmarts(valsartan_smarts)

        sitagliptin_smiles = (
            "NC(CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F"  ### other mol
        )
        sitagliptin_mol = Chem.MolFromSmiles(sitagliptin_smiles)

        target_logp = Descriptors.MolLogP(sitagliptin_mol)
        target_tpsa = Descriptors.TPSA(sitagliptin_mol)
        target_bertz = Descriptors.BertzCT(sitagliptin_mol)

        valsartan_logp_modifier = GaussianModifier(mu=target_logp, sigma=0.2)
        valsartan_tpsa_modifier = GaussianModifier(mu=target_tpsa, sigma=5)
        valsartan_bertz_modifier = GaussianModifier(mu=target_bertz, sigma=30)

    molecule = smiles_to_rdkit_mol(test_smiles)
    matches = molecule.GetSubstructMatches(valsartan_mol)
    if len(matches) > 0:
        smarts_score = 1.0
    else:
        smarts_score = 0.0

    logp_score = valsartan_logp_modifier(Descriptors.MolLogP(molecule))
    tpsa_score = valsartan_tpsa_modifier(Descriptors.TPSA(molecule))
    bertz_score = valsartan_bertz_modifier(Descriptors.BertzCT(molecule))
    valsartan_gmean = gmean([smarts_score, tpsa_score, logp_score, bertz_score])
    return valsartan_gmean


###########################################################################
###               END of Guacamol
###########################################################################
"""
Synthesizability from a full retrosynthetic analysis
Including:
    1. MIT ASKCOS
    ASKCOS (https://askcos.mit.edu) is an open-source software 
    framework that integrates efforts to generalize known chemistry 
    to new substrates by learning to apply retrosynthetic transformations, 
    to identify suitable reaction conditions, and to evaluate whether 
    reactions are likely to be successful. The data-driven models are trained 
    with USPTO and Reaxys databases.
    
    Reference:
    https://doi.org/10.1021/acs.jcim.0c00174

    2. IBM_RXN
    IBM RXN (https://rxn.res.ibm.com) is an AI platform integarting 
    forward reaction prediction and retrosynthetic analysis. The 
    backend of the IBM RXN retrosynthetic analysis is Molecular 
    Transformer model (see reference). The model was mainly trained 
    with USPTO, Pistachio databases.
    Reference:
    https://doi.org/10.1021/acscentsci.9b00576
"""


def tree_analysis(current):
    """
    Analyze the result of tree builder
    Calculate: 1. Number of steps 2. \Pi plausibility 3. If find a path
    In case of celery error, all values are -1

    return:
        num_path = number of paths found
        status: Same as implemented in ASKCOS one
        num_step: number of steps
        p_score: \Pi plausibility
        synthesizability: binary code
        price: price for synthesize query compound
    """
    if "error" in current.keys():
        return -1, {}, 11, -1, -1, -1

    if "price" in current.keys():
        return 0, {}, 0, 1, 1, current["price"]

    num_path = len(current["trees"])
    if num_path != 0:
        current = [current["trees"][0]]
        if current[0]["ppg"] != 0:
            return 0, {}, 0, 1, 1, current[0]["ppg"]
    else:
        current = []

    depth = 0
    p_score = 1
    status = {0: 1}
    price = 0
    while True:
        num_child = 0
        depth += 0.5
        temp = []
        for i, item in enumerate(current):
            num_child += len(item["children"])
            temp = temp + item["children"]
        if num_child == 0:
            break
        if depth % 1 != 0:
            for sth in temp:
                p_score = p_score * sth["plausibility"]
        else:
            for mol in temp:
                price += mol["ppg"]
        status[depth] = num_child
        current = temp
    if len(status) > 1:
        synthesizability = 1
    else:
        synthesizability = 0
    if int(depth - 0.5) == 0:
        depth = 11
        price = -1
    else:
        depth = int(depth - 0.5)
    return num_path, status, depth, p_score * synthesizability, synthesizability, price


def askcos(
    smiles,
    host_ip,
    output="plausibility",
    save_json=False,
    file_name="tree_builder_result.json",
    num_trials=5,
    max_depth=9,
    max_branching=25,
    expansion_time=60,
    max_ppg=100,
    template_count=1000,
    max_cum_prob=0.999,
    chemical_property_logic="none",
    max_chemprop_c=0,
    max_chemprop_n=0,
    max_chemprop_o=0,
    max_chemprop_h=0,
    chemical_popularity_logic="none",
    min_chempop_reactants=5,
    min_chempop_products=5,
    filter_threshold=0.1,
    return_first="true",
):
    """
    The ASKCOS retrosynthetic analysis oracle function.
    Please refer https://github.com/connorcoley/ASKCOS to run the ASKCOS with docker on a server to receive requests.
    """

    if output not in ["num_step", "plausibility", "synthesizability", "price"]:
        raise NameError(
            "This output value is not implemented. Please select one from 'num_step', 'plausibility', 'synthesizability', 'price'."
        )

    import json, requests

    params = {"smiles": smiles}
    resp = requests.get(host_ip + "/api/price/", params=params, verify=False)

    if resp.json()["price"] == 0:
        # Parameters for Tree Builder
        params = {
            "smiles": smiles,
            # optional
            "max_depth": max_depth,
            "max_branching": max_branching,
            "expansion_time": expansion_time,
            "max_ppg": max_ppg,
            "template_count": template_count,
            "max_cum_prob": max_cum_prob,
            "chemical_property_logic": chemical_property_logic,
            "max_chemprop_c": max_chemprop_c,
            "max_chemprop_n": max_chemprop_n,
            "max_chemprop_o": max_chemprop_o,
            "max_chemprop_h": max_chemprop_h,
            "chemical_popularity_logic": chemical_popularity_logic,
            "min_chempop_reactants": min_chempop_reactants,
            "min_chempop_products": min_chempop_products,
            "filter_threshold": filter_threshold,
            "return_first": return_first,
        }

        # For each entry, repeat to test up to num_trials times if got error message
        for _ in range(num_trials):
            print("Trying to send the request, for the %i times now" % (_ + 1))
            resp = requests.get(host_ip + "/api/treebuilder/",
                                params=params,
                                verify=False)
            if "error" not in resp.json().keys():
                break

    if save_json:
        with open(file_name, "w") as f_data:
            json.dump(resp.json(), f_data)

    num_path, status, depth, p_score, synthesizability, price = tree_analysis(
        resp.json())

    if output == "plausibility":
        return p_score
    elif output == "num_step":
        return depth
    elif output == "synthesizability":
        return synthesizability
    elif output == "price":
        return price


def ibm_rxn(smiles, api_key, output="confidence", sleep_time=30):
    """
    This function is modified from Dr. Jan Jensen's code
    """
    try:
        from rxn4chemistry import RXN4ChemistryWrapper
    except:
        print_sys("Please install rxn4chemistry via pip install rxn4chemistry")
    import time

    rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)
    response = rxn4chemistry_wrapper.create_project("test")
    time.sleep(sleep_time)
    response = rxn4chemistry_wrapper.predict_automatic_retrosynthesis(
        product=smiles)
    status = ""
    while status != "SUCCESS":
        time.sleep(sleep_time)
        results = rxn4chemistry_wrapper.get_predict_automatic_retrosynthesis_results(
            response["prediction_id"])
        status = results["status"]

    if output == "confidence":
        return results["retrosynthetic_paths"][0]["confidence"]
    elif output == "result":
        return results
    else:
        raise NameError("This output value is not implemented.")


class molecule_one_retro:

    def __init__(self, api_token):
        try:
            from m1wrapper import MoleculeOneWrapper
        except:
            try:
                install(
                    "git+https://github.com/molecule-one/m1wrapper-python@v1")
                from m1wrapper import MoleculeOneWrapper
            except:
                raise ImportError(
                    "Install Molecule.One Wrapper via pip install git+https://github.com/molecule-one/m1wrapper-python@v1"
                )

        self.m1wrapper = MoleculeOneWrapper(api_token,
                                            "https://tdc.molecule.one")

    def __call__(self, smiles):
        if isinstance(smiles, str):
            smiles = [smiles]

        search = self.m1wrapper.run_batch_search(
            targets=smiles,
            parameters={
                "exploratory_search": False,
                "detail_level": "score"
            },
        )

        status_cur = search.get_status()
        print_sys("Started Querying...")
        print_sys(status_cur)
        while True:
            time.sleep(7)
            status = search.get_status()

            if (status["queued"] == 0) and (status["running"] == 0):
                print_sys("Finished... Returning Results...")
                break
            else:
                if status_cur != status:
                    print_sys(status)
            status_cur = status
        result = search.get_results(precision=5,
                                    only=["targetSmiles", "result"])
        return {i["targetSmiles"]: i["result"] for i in result}


class PyScreener_meta:
    """Evaluate docking score

    Args:

    Return:


    """

    def __init__(
        self,
        receptor_pdb_file,
        box_center,
        box_size,
        software_class="vina",
        ncpu=4,
        **kwargs,
    ):
        try:
            import ray

            try:
                ray.init()
            except:
                ray.shutdown()
                ray.init()
            import pyscreener as ps
        except:
            raise ImportError(
                "Please install PyScreener following guidance in https://github.com/coleygroup/pyscreener"
            )

        try:
            metadata = ps.build_metadata(software_class)
        except:
            raise ValueError(
                'The value of software_class is not implemented. Currently available:["vina", "qvina", "smina", "psovina", "dock", "dock6", "ucsfdock"]'
            )

        self.scorer = ps.virtual_screen(
            software_class,
            [receptor_pdb_file],
            box_center,
            box_size,
            metadata,
            ncpu=ncpu,
        )

    def __call__(self, test_smiles, error_value=None):
        final_score = self.scorer(test_smiles)
        if type(test_smiles) == str:
            return list(final_score)[0]
        else:  ## list
            # dict: {'O=C(/C=C/c1ccc([N+](=O)[O-])o1)c1ccc(-c2ccccc2)cc1': -9.9, 'CCOc1cc(/C=C/C(=O)C(=Cc2ccc(O)c(OC)c2)C(=O)/C=C/c2ccc(O)c(OCC)c2)ccc1O': -9.1}
            # return [list(i.values())[0] for i in final_score]
            score_lst = []
            for smiles in test_smiles:
                score = final_score[smiles]
                if score is None:
                    score = error_value
                score_lst.append(score)
            return score_lst


class Score_3d:
    """Evaluate Vina score (force field) for a conformer binding to a receptor"""

    def __init__(self,
                 receptor_pdbqt_file,
                 center,
                 box_size,
                 scorefunction="vina"):
        try:
            from vina import Vina
        except:
            raise ImportError(
                "Please install vina following guidance in https://github.com/ccsb-scripps/AutoDock-Vina/tree/develop/build/python"
            )

        self.v = Vina(sf_name=scorefunction)
        self.receptor_pdbqt_file = receptor_pdbqt_file
        self.center = center
        self.box_size = box_size
        self.v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt_file)
        try:
            self.v.compute_vina_maps(center=self.center, box_size=self.box_size)
        except:
            raise ValueError(
                "Cannot compute the affinity map, please check center and box_size"
            )

    def __call__(self, ligand_pdbqt_file, minimize=True):
        try:
            self.v.set_ligand_from_file(ligand_pdbqt_file)
            if minimize:
                energy = self.v.optimize()[0]
            else:
                energy = self.v.score()[0]
        except Exception as e:
            print(e)
            return np.inf
        return energy


class Vina_3d:
    """Perform docking search from a conformer."""

    def __init__(self,
                 receptor_pdbqt_file,
                 center,
                 box_size,
                 scorefunction="vina"):
        try:
            from vina import Vina
        except:
            raise ImportError(
                "Please install vina following guidance in https://github.com/ccsb-scripps/AutoDock-Vina/tree/develop/build/python"
            )

        self.v = Vina(sf_name=scorefunction)
        self.receptor_pdbqt_file = receptor_pdbqt_file
        self.center = center
        self.box_size = box_size
        self.v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt_file)
        try:
            self.v.compute_vina_maps(center=self.center, box_size=self.box_size)
        except:
            raise ValueError(
                "Cannot compute the affinity map, please check center and box_size"
            )

    def __call__(self,
                 ligand_pdbqt_file,
                 output_file="out.pdbqt",
                 exhaustiveness=8,
                 n_poses=10):
        try:
            self.v.set_ligand_from_file(ligand_pdbqt_file)
            self.v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            self.v.write_poses(output_file, n_poses=n_poses, overwrite=True)
            energy = self.v.score()[0]
        except Exception as e:
            print(e)
            return np.inf
        return energy


class Vina_smiles:
    """Perform docking search from a conformer."""

    def __init__(self,
                 receptor_pdbqt_file,
                 center,
                 box_size,
                 scorefunction="vina"):
        try:
            from vina import Vina
        except:
            raise ImportError(
                "Please install vina following guidance in https://github.com/ccsb-scripps/AutoDock-Vina/tree/develop/build/python"
            )

        self.v = Vina(sf_name=scorefunction)
        self.receptor_pdbqt_file = receptor_pdbqt_file
        self.center = center
        self.box_size = box_size
        self.v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt_file)
        # self.v.set_receptor(rigid_pdbqt_filename = "tdc/receptors/1iep/1iep_receptor.pdbqt")
        try:
            self.v.compute_vina_maps(center=self.center, box_size=self.box_size)
        except:
            raise ValueError(
                "Cannot compute the affinity map, please check center and box_size"
            )

    def __call__(self,
                 ligand_smiles,
                 output_file="out.pdbqt",
                 exhaustiveness=8,
                 n_poses=10):
        try:
            m = Chem.MolFromSmiles(ligand_smiles)
            m = Chem.AddHs(m)
            AllChem.EmbedMolecule(m)
            AllChem.MMFFOptimizeMolecule(m)
            print(Chem.MolToMolBlock(m), file=open("__temp.mol", "w+"))
            os.system("mk_prepare_ligand.py -i __temp.mol -o __temp.pdbqt")
            self.v.set_ligand_from_file("__temp.pdbqt")
            self.v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            self.v.write_poses(output_file, n_poses=n_poses, overwrite=True)
            energy = self.v.score()[0]
            os.system("rm __temp.mol __temp.pdbqt")
        except Exception as e:
            print(e)
            return np.inf
        return energy


def smina(ligand, protein, score_only=False, raw_input=False):
    """
    Sima is a docking algorithm that docks a ligand to a protein pocket.

    Koes, D.R., Baumgartner, M.P. and Camacho, C.J., 2013.
    Lessons learned in empirical scoring with smina from the CSAR 2011 benchmarking exercise.
    Journal of chemical information and modeling, 53(8), pp.1893-1904.

    Parameters
    ----------
    ligand : array
        (N_1,3) matrix, where N_1 is ligand size.
    protein : array
        (N_2,3) matrix, where N_2 is protein size.
    score_only: boolean
        whether to only return docking score.
    raw_input: boolean
        whether to input raw ML input or sdf file input
    Returns
    -------
    docking_info: str or float
        docking result

    """
    smina_model_path = "oracle/smina.static"
    os.system(f"chmod +x ./{smina_model_path}")
    # if machine learning raw input:
    # 1. write out to xyz file
    # 2. convert to sdf file by openbabel
    if raw_input:
        mol_coord, mol_atom = ligand
        # 1. write out to xyz file
        f = open(f"temp_ligand.xyz", "w")
        n_atoms = len(mol_atom)
        f.write("%d\n\n" % n_atoms)
        for atom_i in range(n_atoms):
            atom = mol_atom[atom_i]
            f.write("%s %.9f %.9f %.9f\n" % (
                atom,
                mol_coord[atom_i, 0],
                mol_coord[atom_i, 1],
                mol_coord[atom_i, 2],
            ))
        f.close()
        # 2. convert to sdf file
        try:
            os.system(f"obabel temp_ligand.xyz -O temp_ligand.sdf")
        except:
            raise ImportError(
                "Please install openbabel by 'conda install -c conda-forge openbabel'!"
            )
        ligand = "temp_ligand.sdf"
    if score_only:
        msg = os.popen(
            f"./{smina_model_path} -l {ligand} -r {protein} --score_only").read(
            )
        return float(msg.split("\n")[-7].split(" ")[-2])
    else:
        os.system(f"./{smina_model_path} -l {ligand} -r {protein} --score_only")


# os.system("python docking.py " + ligand_pdbqt_file + \
#           " "+target_pdbqt_file + " " + output_file +' '+ \
#           docking_center_string + ' ' + box_size_string)
