import pickle 
import numpy as np 
import re
import os.path as op
import math
from collections import defaultdict
from abc import abstractmethod
from functools import partial
from typing import List

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

from .utils import oracle_load




'''
copy from https://benevolent.ai/guacamol.

'''


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
        return 1. - np.abs(self.target_value - x)


class GaussianModifier(ScoreModifier):
    """
    Score modifier that reproduces a Gaussian bell shape.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2.))


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

    The function looks like this:

       upper_x < lower_x                 lower_x < upper_x
    __________                                   ____________
              \                                 /
               \                               /
                \__________          _________/

    This class works as follows:
    First the input is mapped onto a linear interpolation between both specified points.
    Then the generated values are clipped between low and high scores.
    """

    def __init__(self, upper_x: float, lower_x=0.0, high_score=1.0, low_score=0.0) -> None:
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

    def __init__(self, upper_x: float, lower_x=0.0, high_score=1.0, low_score=0.0) -> None:
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
        return self.low_score + self.L / (1 + np.exp(-self.k * (x - self.middle_x)))


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
def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    # if name == "fpscores":
    #     name = op.join(previous_directory(op.dirname(__file__)), name)
    name = oracle_load('fpscores')
    with open('oracle/fpscores.pkl', "rb") as f:
      _fscores = pickle.load(f)
    outDict = {}
    for i in _fscores:
        for j in range(1,len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol,ri=None):
  nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
  nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
  return nBridgehead,nSpiro

def calculateScore(m):
  if _fscores is None: readFragmentScores()

  # fragment score
  fp = rdMolDescriptors.GetMorganFingerprint(m,2)  #<- 2 is the *radius* of the circular fingerprint
  fps = fp.GetNonzeroElements()
  score1 = 0.
  nf = 0
  for bitId,v in iteritems(fps):
    nf += v
    sfp = bitId
    score1 += _fscores.get(sfp,-4)*v
  score1 /= nf

  # features score
  nAtoms = m.GetNumAtoms()
  nChiralCenters = len(Chem.FindMolChiralCenters(m,includeUnassigned=True))
  ri = m.GetRingInfo()
  nBridgeheads,nSpiro=numBridgeheadsAndSpiro(m,ri)
  nMacrocycles=0
  for x in ri.AtomRings():
    if len(x)>8: nMacrocycles+=1

  sizePenalty = nAtoms**1.005 - nAtoms
  stereoPenalty = math.log10(nChiralCenters+1)
  spiroPenalty = math.log10(nSpiro+1)
  bridgePenalty = math.log10(nBridgeheads+1)
  macrocyclePenalty = 0.
  # ---------------------------------------
  # This differs from the paper, which defines:
  #  macrocyclePenalty = math.log10(nMacrocycles+1)
  # This form generates better results when 2 or more macrocycles are present
  if nMacrocycles > 0: macrocyclePenalty = math.log10(2)

  score2 = 0. -sizePenalty -stereoPenalty -spiroPenalty -bridgePenalty -macrocyclePenalty

  # correction for the fingerprint density
  # not in the original publication, added in version 1.1
  # to make highly symmetrical molecules easier to synthetise
  score3 = 0.
  if nAtoms > len(fps):
    score3 = math.log(float(nAtoms) / len(fps)) * .5

  sascore = score1 + score2 + score3

  # need to transform "raw" value into scale between 1 and 10
  min = -4.0
  max = 2.5
  sascore = 11. - (sascore - min + 1) / (max - min) * 9.
  # smooth the 10-end
  if sascore > 8.: sascore = 8. + math.log(sascore+1.-9.)
  if sascore > 10.: sascore = 10.0
  elif sascore < 1.: sascore = 1.0 

  return sascore



"""Scores based on an ECFP classifier for activity."""

# clf_model = None
def load_drd2_model():
    # global clf_model
    # name = op.join(op.dirname(__file__), 'clf_py36.pkl')
    print("==== load drd2 oracle =====")
    name = 'oracle/drd2.pkl'
    with open(name, "rb") as f:
        clf_model = pickle.load(f)
    return clf_model

def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp


def drd2(smile):

    if 'drd2_model' not in globals().keys():
        global drd2_model
        drd2_model = load_drd2_model() 

    mol = Chem.MolFromSmiles(smile)
    if mol:
        fp = fingerprints_from_mol(mol)
        score = drd2_model.predict_proba(fp)[:, 1]
        return float(score)
    return 0.0






## from https://github.com/wengong-jin/iclr19-graph2graph/blob/master/props/properties.py 
## from https://github.com/wengong-jin/multiobj-rationale/blob/master/properties.py 

def similarity(a, b):
	if a is None or b is None: 
		return 0.0
	amol = Chem.MolFromSmiles(a)
	bmol = Chem.MolFromSmiles(b)
	if amol is None or bmol is None:
		return 0.0
	fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
	fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
	return DataStructs.TanimotoSimilarity(fp1, fp2) 

def qed(s):
	if s is None: 
		return 0.0
	mol = Chem.MolFromSmiles(s)
	if mol is None: 
		return 0.0
	return QED.qed(mol)

def penalized_logp(s):
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
	if s is None:
		return 100 
	mol = Chem.MolFromSmiles(s)
	if mol is None:
		return 100 
	SAscore = calculateScore(mol)
	return SAscore 	

'''
for gsk3 and jnk3, 
some code are borrowed from 
https://github.com/wengong-jin/multiobj-rationale/blob/master/properties.py 

'''

# class gsk3b:
# 	def __init__(self):
# 		gsk3_model_path = 'oracle/gsk3b.pkl'
# 		with open(gsk3_model_path, 'rb') as f:
# 			self.gsk3_model = pickle.load(f)

# 	def __call__(self, smiles):
# 		molecule = smiles_to_rdkit_mol(smiles)
# 		fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
# 		features = np.zeros((1,))
# 		DataStructs.ConvertToNumpyArray(fp, features)
# 		fp = features.reshape(1, -1) 
# 		gsk3_score = self.gsk3_model.predict_proba(fp)[0,1]
# 		return gsk3_score 

def load_gsk3b_model():
    gsk3_model_path = 'oracle/gsk3b.pkl'
    print('==== load gsk3b oracle =====')
    with open(gsk3_model_path, 'rb') as f:
        gsk3_model = pickle.load(f)
    return gsk3_model 

def gsk3b(smiles):
    if 'gsk3_model' not in globals().keys():
        global gsk3_model 
        gsk3_model = load_gsk3b_model()

    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, features)
    fp = features.reshape(1, -1) 
    gsk3_score = gsk3_model.predict_proba(fp)[0,1]
    return gsk3_score 


class jnk3:
	def __init__(self):
		jnk3_model_path = 'oracle/jnk3.pkl'
		with open(jnk3_model_path, 'rb') as f:
			self.jnk3_model = pickle.load(f)

	def __call__(self, smiles):
		molecule = smiles_to_rdkit_mol(smiles)
		fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
		features = np.zeros((1,))
		DataStructs.ConvertToNumpyArray(fp, features)
		fp = features.reshape(1, -1) 
		jnk3_score = self.jnk3_model.predict_proba(fp)[0,1]
		return jnk3_score 	

def single_molecule_validity(smiles):
	if smiles.strip() == '':
		return False 
	mol = Chem.MolFromSmiles(smiles)
	if mol is None or mol.GetNumAtoms() == 0:
		return False 
	return True

def validity_ratio(list_of_smiles):
	valid_list_smiles = list(filter(single_molecule_validity, list_of_smiles))
	return 1.0*len(valid_list_smiles)/len(list_of_smiles)


def canonicalize(smiles):
	mol = Chem.MolFromSmiles(smiles)
	if mol is not None:
		return Chem.MolToSmiles(mol, isomericSmiles=True)
	else:
		return None

def unique_lst_of_smiles(list_of_smiles):
	canonical_smiles_lst = list(map(canonicalize, list_of_smiles))
	canonical_smiles_lst = list(filter(lambda x:x is not None, canonical_smiles_lst))
	canonical_smiles_lst = list(set(canonical_smiles_lst))
	return canonical_smiles_lst

def unique_rate(list_of_smiles):
	canonical_smiles_lst = unique_lst_of_smiles(list_of_smiles)
	return 1.0*len(canonical_smiles_lst)/len(list_of_smiles)

def novelty(new_smiles, smiles_database):
	new_smiles = unique_lst_of_smiles(new_smiles)
	smiles_database = unique_lst_of_smiles(smiles_database)
	novel_ratio = sum([1 if i in smiles_database else 0 for i in new_smiles])*1.0 / len(new_smiles)
	return 1 - novel_ratio

def diversity(list_of_smiles):
	"""
		The diversity of a set of molecules is defined as the average pairwise
		Tanimoto distance between the Morgan fingerprints ---- GCPN
	"""
	list_of_unique_smiles = unique_lst_of_smiles(list_of_smiles)
	list_of_mol = [Chem.MolFromSmiles(smiles) for smiles in list_of_unique_smiles]
	list_of_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False) for mol in list_of_mol]
	avg_lst = []
	for idx, fp in enumerate(list_of_fp):
		for fp2 in list_of_fp[idx+1:]:
			sim = DataStructs.TanimotoSimilarity(fp, fp2) 			
			avg_lst.append(sim)
	return np.mean(avg_lst)

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


def celecoxib_rediscovery(test_smiles):
	# celecoxib_smiles = 'CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F'
	# 'ECFP4'

  if 'celecoxib_fp' not in globals().keys():
    global celecoxib_fp
    celecoxib_smiles = 'CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F'
    celecoxib_fp = smiles_2_fingerprint_ECFP4(celecoxib_smiles)

	test_fp = smiles_2_fingerprint_ECFP4(test_smiles)
	similarity_value = DataStructs.TanimotoSimilarity(celecoxib_fp, test_fp)
	return similarity_value


def troglitazone_rediscovery(test_smiles):
	### ECFP4

  if 'troglitazone_fp' not in globals().keys():
    global troglitazone_fp
    troglitazone_smiles='Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O'
    troglitazone_fp = smiles_2_fingerprint_ECFP4(troglitazone_smiles)

	test_fp = smiles_2_fingerprint_ECFP4(test_smiles)
	similarity_value = DataStructs.TanimotoSimilarity(troglitazone_fp, test_fp)
	return similarity_value	


def thiothixene_rediscovery(test_smiles):
	### ECFP4

  if 'Thiothixene_fp' not in globals().keys():
    global Thiothixene_fp
    Thiothixene_smiles='CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1'  
    Thiothixene_fp = smiles_2_fingerprint_ECFP4(Thiothixene_smiles)

	test_fp = smiles_2_fingerprint_ECFP4(test_smiles)
	similarity_value = DataStructs.TanimotoSimilarity(Thiothixene_fp, test_fp)
	return similarity_value




def aripiprazole_similarity(test_smiles):
	threshold = 0.75
  if 'Aripiprazole_fp' not in globals().keys():
    global Aripiprazole_fp
    Aripiprazole_smiles = 'Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl'
    Aripiprazole_fp = smiles_2_fingerprint_FCFP4(Aripiprazole_smiles)

	test_fp = smiles_2_fingerprint_FCFP4(test_smiles)
	similarity_value = DataStructs.TanimotoSimilarity(Aripiprazole_fp, test_fp)
	modifier = ClippedScoreModifier(upper_x=threshold)
	modified_similarity = modifier(similarity_value)
	return modified_similarity 


def albuterol_similarity(test_smiles):
	threshold = 0.75
  if 'Albuterol_fp' not in globals().keys():
    global Albuterol_fp
    Albuterol_smiles = 'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1'
    Albuterol_fp = smiles_2_fingerprint_FCFP4(Albuterol_smiles)

	test_fp = smiles_2_fingerprint_FCFP4(test_smiles)
	similarity_value = DataStructs.TanimotoSimilarity(Albuterol_fp, test_fp)
	modifier = ClippedScoreModifier(upper_x=threshold)
	modified_similarity = modifier(similarity_value)
	return modified_similarity 




def mestranol_similarity(test_smiles):
	threshold = 0.75 
  if 'Mestranol_fp' not in globals().keys():
    global Mestranol_fp
    Mestranol_smiles = 'COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1'
    Mestranol_fp = smiles_2_fingerprint_AP(Mestranol_smiles)

	test_fp = smiles_2_fingerprint_AP(test_smiles)
	similarity_value = DataStructs.TanimotoSimilarity(Mestranol_fp, test_fp)
	modifier = ClippedScoreModifier(upper_x=threshold)
	modified_similarity = modifier(similarity_value)
	return modified_similarity 


def median1(test_smiles):
	# median mol between camphor and menthol, ECFP4 
  if 'camphor_fp' not in globals().keys():
    global camphor_fp, menthol_fp
    camphor_smiles = 'CC1(C)C2CCC1(C)C(=O)C2'
    menthol_smiles = 'CC(C)C1CCC(C)CC1O'
    camphor_fp = smiles_2_fingerprint_ECFP4(camphor_smiles)
    menthol_fp = smiles_2_fingerprint_ECFP4(menthol_smiles)
	
  test_fp = smiles_2_fingerprint_ECFP4(test_smiles)

	similarity_v1 = DataStructs.TanimotoSimilarity(camphor_fp, test_fp)
	similarity_v2 = DataStructs.TanimotoSimilarity(menthol_fp, test_fp)
	similarity_gmean = gmean([similarity_v1, similarity_v2])
	return similarity_gmean



def median2(test_smiles):
	# median mol between tadalafil and sildenafil, ECFP6 

  if 'tadalafil_fp' not in globals().keys():
    global tadalafil_fp, sildenafil_fp
    tadalafil_smiles = 'O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C'
    sildenafil_smiles = 'CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C'
    tadalafil_fp = smiles_2_fingerprint_ECFP6(tadalafil_smiles)
    sildenafil_fp = smiles_2_fingerprint_ECFP6(sildenafil_smiles)
	
  test_fp = smiles_2_fingerprint_ECFP6(test_smiles)
	similarity_v1 = DataStructs.TanimotoSimilarity(tadalafil_fp, test_fp)
	similarity_v2 = DataStructs.TanimotoSimilarity(sildenafil_fp, test_fp)
	similarity_gmean = gmean([similarity_v1, similarity_v2])
	return similarity_gmean 









osimertinib_smiles = 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34'
osimertinib_fp_fcfc4 = smiles_2_fingerprint_FCFP4(osimertinib_smiles)
osimertinib_fp_ecfc6 = smiles_2_fingerprint_ECFP6(osimertinib_smiles)

def osimertinib_mpo(test_smiles):

	sim_v1_modifier = ClippedScoreModifier(upper_x=0.8)
	sim_v2_modifier = MinGaussianModifier(mu=0.85, sigma=0.1)
	tpsa_modifier = MaxGaussianModifier(mu=100, sigma=10) 
	logp_modifier = MinGaussianModifier(mu=1, sigma=1) 

	molecule = smiles_to_rdkit_mol(test_smiles)
	fp_fcfc4 = smiles_2_fingerprint_FCFP4(test_smiles)
	fp_ecfc6 = smiles_2_fingerprint_ECFP6(test_smiles)
	tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
	logp_score = logp_modifier(Descriptors.MolLogP(molecule))
	similarity_v1 = sim_v1_modifier(DataStructs.TanimotoSimilarity(osimertinib_fp_fcfc4, fp_fcfc4))
	similarity_v2 = sim_v2_modifier(DataStructs.TanimotoSimilarity(osimertinib_fp_ecfc6, fp_ecfc6))

	osimertinib_gmean = gmean([tpsa_score, logp_score, similarity_v1, similarity_v2])
	return osimertinib_gmean 

fexofenadine_smiles = 'CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4'
fexofenadine_fp = smiles_2_fingerprint_AP(fexofenadine_smiles)
def Fexofenadine_mpo(test_smiles):
	similar_modifier = ClippedScoreModifier(upper_x=0.8)
	tpsa_modifier=MaxGaussianModifier(mu=90, sigma=10)
	logp_modifier=MinGaussianModifier(mu=4, sigma=1)

	molecule = smiles_to_rdkit_mol(test_smiles)
	fp_ap = smiles_2_fingerprint_AP(test_smiles)
	tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
	logp_score = logp_modifier(Descriptors.MolLogP(molecule))
	similarity_value = similar_modifier(DataStructs.TanimotoSimilarity(fp_ap, fexofenadine_fp))
	fexofenadine_gmean = gmean([tpsa_score, logp_score, similarity_value])
	return fexofenadine_gmean 









class AtomCounter:

    def __init__(self, element: str) -> None:
        """
        Args:
            element: element to count within a molecule
        """
        self.element = element

    def __call__(self, mol: Mol) -> int:
        """
        Count the number of atoms of a given type.

        Args:
            mol: molecule

        Returns:
            The number of atoms of the given type.
        """
        # if the molecule contains H atoms, they may be implicit, so add them
        if self.element == 'H':
            mol = Chem.AddHs(mol)

        return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == self.element)

ranolazine_smiles = 'COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2'
ranolazine_fp = smiles_2_fingerprint_AP(ranolazine_smiles)
fluorine_counter = AtomCounter('F')
def Ranolazine_mpo(test_smiles):
  
  similar_modifier = ClippedScoreModifier(upper_x=0.7)
  tpsa_modifier = MaxGaussianModifier(mu=95, sigma=20)
  logp_modifier = MaxGaussianModifier(mu=7, sigma=1)
  fluorine_modifier = GaussianModifier(mu=1, sigma=1.0)

  molecule = smiles_to_rdkit_mol(test_smiles)
  fp_ap = smiles_2_fingerprint_AP(test_smiles)
  tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
  logp_score = logp_modifier(Descriptors.MolLogP(molecule))
  similarity_value = similar_modifier(DataStructs.TanimotoSimilarity(fp_ap, ranolazine_fp))
  fluorine_value = fluorine_modifier(fluorine_counter(molecule))

  ranolazine_gmean = gmean([tpsa_score, logp_score, similarity_value, fluorine_value])
  return ranolazine_gmean








perindopril_smiles = 'O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC'
perindopril_fp = smiles_2_fingerprint_ECFP4(perindopril_smiles)
def num_aromatic_rings(mol):
    return rdMolDescriptors.CalcNumAromaticRings(mol)

def Perindopril_mpo(test_smiles):
  ## no similar_modifier
  arom_rings_modifier = GaussianModifier(mu = 2, sigma = 0.5)

  molecule = smiles_to_rdkit_mol(test_smiles)
  fp_ecfp4 = smiles_2_fingerprint_ECFP4(test_smiles)

  similarity_value = DataStructs.Tanimoto(fp_ecfp4, perindopril_fp)
  num_aromatic_rings_value = arom_rings_modifier(num_aromatic_rings(molecule))

  perindopril_gmean = gmean([similarity_value, num_aromatic_rings_value])
  return perindopril_gmean







amlodipine_smiles = 'Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC'
amlodipine_fp = smiles_2_fingerprint_ECFP4(amlodipine_smiles)
def num_rings(mol):
    return rdMolDescriptors.CalcNumRings(mol)

def Amlodipine_mpo(test_smiles):
  ## no similar_modifier
  num_rings_modifier = GaussianModifier(mu=3, sigma=0.5)

  molecule = smiles_to_rdkit_mol(test_smiles)
  fp_ecfp4 = smiles_2_fingerprint_ECFP4(test_smiles)

  similarity_value = DataStructs.Tanimoto(fp_ecfp4, amlodipine_fp)
  num_rings_value = arom_rings_modifier(num_rings(molecule))

  amlodipine_gmean = gmean([similarity_value, num_rings_value])
  return amlodipine_gmean
















###########################################################################
###               END of Guacamol
###########################################################################


'''
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
'''

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
    if 'error' in current.keys():
        return -1, {}, 11, -1, -1, -1
    
    if 'price' in current.keys():
        return 0, {}, 0, 1, 1, current['price']
    
    num_path = len(current['trees'])
    if num_path != 0:
        current = [current['trees'][0]]
        if current[0]['ppg'] != 0:
            return 0, {}, 0, 1, 1, current[0]['ppg']
    else:
        current = []
        
    depth = 0
    p_score = 1
    status = {0:1}
    price = 0
    while True:
        num_child = 0
        depth += 0.5
        temp = []
        for i, item in enumerate(current):
            num_child += len(item['children'])
            temp = temp + item['children']
        if num_child == 0:
            break
        if depth % 1 != 0:
            for sth in temp:
                p_score = p_score * sth['plausibility']
        else:
            for mol in temp:
                price += mol['ppg']
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
    return num_path, status, depth, p_score*synthesizability, synthesizability, price

def askcos(smiles, host_ip, output='plausibility', save_json=False, file_name='tree_builder_result.json', num_trials=5,
           max_depth=9, max_branching=25, expansion_time=60, max_ppg=100, template_count=1000, max_cum_prob=0.999, 
           chemical_property_logic='none', max_chemprop_c=0, max_chemprop_n=0, max_chemprop_o=0, max_chemprop_h=0, 
           chemical_popularity_logic='none', min_chempop_reactants=5, min_chempop_products=5, filter_threshold=0.1, return_first='true'):
    """
    The ASKCOS retrosynthetic analysis oracle function. 
    Please refer https://github.com/connorcoley/ASKCOS to run the ASKCOS with docker on a server to receive requests.
    """

    if output not in ['num_step', 'plausibility', 'synthesizability', 'price']:
        raise NameError("This output value is not implemented. Please select one from 'num_step', 'plausibility', 'synthesizability', 'price'.")
    
    import json, requests
    
    params = {
        'smiles': smiles
    }
    resp = requests.get(host_ip+'/api/price/', params=params, verify=False)

    if resp.json()['price'] == 0:
        # Parameters for Tree Builder
        params = {
            'smiles': smiles, 

            # optional
            'max_depth': max_depth,
            'max_branching': max_branching,
            'expansion_time': expansion_time,
            'max_ppg': max_ppg,
            'template_count': template_count,
            'max_cum_prob': max_cum_prob,
            'chemical_property_logic': chemical_property_logic,
            'max_chemprop_c': max_chemprop_c,
            'max_chemprop_n': max_chemprop_n,
            'max_chemprop_o': max_chemprop_o,
            'max_chemprop_h': max_chemprop_h,
            'chemical_popularity_logic': chemical_popularity_logic,
            'min_chempop_reactants': min_chempop_reactants,
            'min_chempop_products': min_chempop_products,
            'filter_threshold': filter_threshold,
            'return_first': return_first
        }

        # For each entry, repeat to test up to num_trials times if got error message
        for _ in range(num_trials):
            print('Trying to send the request, for the %i times now' % (_ + 1))
            resp = requests.get(HOST + '/api/treebuilder/', params=params, verify=False)
            if 'error' not in resp.json().keys():
                break
                
    if save_json:
        with open(name, 'w') as f_data:
            json.dump(resp.json(), f_data)
        
    num_path, status, depth, p_score, synthesizability, price = tree_analysis(resp.json())
    
    if output == 'plausibility':
        return p_score
    elif output == 'num_step':
        return depth
    elif output == 'synthesizability':
        return synthesizability
    elif output == 'price':
        return price

def ibm_rxn(smiles, api_key, output='confidence', sleep_time=30):
    
    from rxn4chemistry import RXN4ChemistryWrapper
    import time
    
    rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)
    response = rxn4chemistry_wrapper.create_project('test')
    time.sleep(sleep_time)
    response = rxn4chemistry_wrapper.predict_automatic_retrosynthesis(product=smiles)
    status = ''
    while status != 'SUCCESS':
        time.sleep(sleep_time)
        results = rxn4chemistry_wrapper.get_predict_automatic_retrosynthesis_results(response['prediction_id'])
        status = results['status']

    if output == 'confidence':
        return results['retrosynthetic_paths'][0]['confidence']
    elif output == 'result':
        return results
    else:
        raise NameError("This output value is not implemented.")


'''
    def get_AP(self, mol: Mol):
        return AllChem.GetAtomPairFingerprint(mol, maxLength=10)

    def get_PHCO(self, mol: Mol):
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

    def get_BPF(self, mol: Mol):
        return GetBPFingerprint(mol)

    def get_BTF(self, mol: Mol):
        return GetBTFingerprint(mol)

    def get_PATH(self, mol: Mol):
        return AllChem.RDKFingerprint(mol)

    def get_ECFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2)

    def get_ECFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3)

    def get_FCFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

    def get_FCFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)

'''




if __name__ == "__main__":
	smiles = '[H][C@@]12C[C@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'
	smiles = 'CCC'
	smiles = '[NH3+][C@H](Cc1ccc(F)cc1)[C@H](O)C(=O)[O-]'
	smiles = 'c1ccc(-c2cnc(SC3CCCC3)n2Cc2ccco2)cc1'
	# print(similarity(smiles, smiles))
	# print(qed(smiles))
	# print(penalized_logp(smiles))
	print(drd2(smiles))
	# print(SA(smiles))
	# list_of_smiles = ['CCC', 'fewjio', smiles, smiles]
	# print(validity_ratio(list_of_smiles))
	# print(unique_rate(list_of_smiles))
	# #  conda install -c rdkit rdkit
	# print(Mestranol_similarity(smiles))
	# print(median1(smiles))
	# print(median2(smiles))
	# print(osimertinib_mpo(smiles))
	print(gsk3(smiles))
	print(jnk3(smiles))
	# print(Fexofenadine_mpo(smiles))




