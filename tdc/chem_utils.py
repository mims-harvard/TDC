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

from .utils import oracle_load,print_sys



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
    try:
      with open('oracle/fpscores.pkl', "rb") as f:
        _fscores = pickle.load(f)
    except EOFError:
      import sys
      sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")

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
    #print("==== load drd2 oracle =====")
    name = 'oracle/drd2.pkl'
    try:
      with open(name, "rb") as f:
          clf_model = pickle.load(f)
    except EOFError:
      import sys
      sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")

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
    #print_sys('==== load gsk3b oracle =====')
    try:
      with open(gsk3_model_path, 'rb') as f:
          gsk3_model = pickle.load(f)
    except EOFError:
      import sys
      sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")
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
    try:
      with open(jnk3_model_path, 'rb') as f:
        self.jnk3_model = pickle.load(f)
    except EOFError:
      import sys
      sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")
  
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

def validity(list_of_smiles):
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

def uniqueness(list_of_smiles):
	canonical_smiles_lst = unique_lst_of_smiles(list_of_smiles)
	return 1.0*len(canonical_smiles_lst)/len(list_of_smiles)

def novelty(generated_smiles_lst, training_smiles_lst):
	generated_smiles_lst = unique_lst_of_smiles(generated_smiles_lst)
	training_smiles_lst = unique_lst_of_smiles(training_smiles_lst)
	novel_ratio = sum([1 if i in training_smiles_lst else 0 for i in generated_smiles_lst])*1.0 / len(generated_smiles_lst)
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



########################################
######## KL divergence ########

def _calculate_pc_descriptors(smiles, pc_descriptors):
    from rdkit.ML.Descriptors import MoleculeDescriptors
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(pc_descriptors)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    _fp = calc.CalcDescriptors(mol)
    _fp = np.array(_fp)
    mask = np.isfinite(_fp)
    if (mask == 0).sum() > 0:
        logger.warning(f'{smiles} contains an NAN physchem descriptor')
        _fp[~mask] = 0

    return _fp 

def calculate_pc_descriptors(smiles, pc_descriptors):
    output = []

    for i in smiles:
        d = _calculate_pc_descriptors(i, pc_descriptors)
        if d is not None:
            output.append(d)

    return np.array(output)


def continuous_kldiv(X_baseline: np.array, X_sampled: np.array) -> float:
    from scipy.stats import entropy, gaussian_kde
    kde_P = gaussian_kde(X_baseline)
    kde_Q = gaussian_kde(X_sampled)
    x_eval = np.linspace(np.hstack([X_baseline, X_sampled]).min(), np.hstack([X_baseline, X_sampled]).max(), num=1000)
    P = kde_P(x_eval) + 1e-10
    Q = kde_Q(x_eval) + 1e-10

    return entropy(P, Q)

def discrete_kldiv(X_baseline: np.array, X_sampled: np.array) -> float:
    from scipy.stats import entropy
    from scipy import histogram
    P, bins = histogram(X_baseline, bins=10, density=True)
    P += 1e-10
    Q, _ = histogram(X_sampled, bins=bins, density=True)
    Q += 1e-10

    return entropy(P, Q)

def get_fingerprints(mols, radius=2, length=4096):
    """
    Converts molecules to ECFP bitvectors.

    Args:
        mols: RDKit molecules
        radius: ECFP fingerprint radius
        length: number of bits

    Returns: a list of fingerprints
    """
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, length) for m in mols]


def get_mols(smiles_list):
    for i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(i)
            if mol is not None:
                yield mol
        except Exception as e:
            logger.warning(e)



def calculate_internal_pairwise_similarities(smiles_list):
    """
    Computes the pairwise similarities of the provided list of smiles against itself.

    Returns:
        Symmetric matrix of pairwise similarities. Diagonal is set to zero.
    """
    if len(smiles_list) > 10000:
        logger.warning(f'Calculating internal similarity on large set of '
                       f'SMILES strings ({len(smiles_list)})')

    mols = get_mols(smiles_list)
    fps = get_fingerprints(mols)
    nfps = len(fps)

    similarities = np.zeros((nfps, nfps))

    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        similarities[i, :i] = sims
        similarities[:i, i] = sims

    return similarities



def kl_divergence(generated_smiles_lst, training_smiles_lst):
  pc_descriptor_subset = [
            'BertzCT',
            'MolLogP',
            'MolWt',
            'TPSA',
            'NumHAcceptors',
            'NumHDonors',
            'NumRotatableBonds',
            'NumAliphaticRings',
            'NumAromaticRings'
  ]

  def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True) ### todo double check
    else:
        return None


  generated_lst_mol = list(map(canonical, generated_smiles_lst))
  training_lst_mol = list(map(canonical, training_smiles_lst))
  filter_out_func = lambda x:x is not None 
  generated_lst_mol = list(filter(filter_out_func, generated_lst_mol))
  training_lst_mol = list(filter(filter_out_func, generated_lst_mol))

  d_sampled = calculate_pc_descriptors(generated_lst_mol, pc_descriptor_subset)
  d_chembl = calculate_pc_descriptors(training_lst_mol, pc_descriptor_subset)

  kldivs = {}
  for i in range(4):
    kldiv = continuous_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
    kldivs[pc_descriptor_subset[i]] = kldiv

  # ... and for the int valued ones.
  for i in range(4, 9):
    kldiv = discrete_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
    kldivs[pc_descriptor_subset[i]] = kldiv


  # pairwise similarity

  chembl_sim = calculate_internal_pairwise_similarities(training_lst_mol)
  chembl_sim = chembl_sim.max(axis=1)

  sampled_sim = calculate_internal_pairwise_similarities(generated_lst_mol)
  sampled_sim = sampled_sim.max(axis=1)

  kldiv_int_int = continuous_kldiv(X_baseline=chembl_sim, X_sampled=sampled_sim)
  kldivs['internal_similarity'] = kldiv_int_int
  '''
        # for some reason, this runs into problems when both sets are identical.
        # cross_set_sim = calculate_pairwise_similarities(self.training_set_molecules, unique_molecules)
        # cross_set_sim = cross_set_sim.max(axis=1)
        #
        # kldiv_ext = discrete_kldiv(chembl_sim, cross_set_sim)
        # kldivs['external_similarity'] = kldiv_ext
        # kldiv_sum += kldiv_ext
  '''

  # Each KL divergence value is transformed to be in [0, 1].
  # Then their average delivers the final score.
  partial_scores = [np.exp(-score) for score in kldivs.values()]
  score = sum(partial_scores) / len(partial_scores)
  return score 



def fcd_distance_tf(generated_smiles_lst, training_smiles_lst):
  import pkgutil, tempfile, os
  if 'chemnet' not in globals().keys():
    global chemnet
    ### _load_chemnet
    chemnet_model_filename='ChemNet_v0.13_pretrained.h5'
    model_bytes = pkgutil.get_data('fcd', chemnet_model_filename)
    tmpdir = tempfile.gettempdir()
    model_path = os.path.join(tmpdir, chemnet_model_filename)
    with open(model_path, 'wb') as f:
      f.write(model_bytes)
    chemnet = fcd.load_ref_model(model_path)
    # _load_chemnet

  def _calculate_distribution_statistics(chemnet, molecules):
    sample_std = fcd.canonical_smiles(molecules)
    gen_mol_act = fcd.get_predictions(chemnet, sample_std)

    mu = np.mean(gen_mol_act, axis=0)
    cov = np.cov(gen_mol_act.T)
    return mu, cov

  mu_ref, cov_ref = _calculate_distribution_statistics(chemnet, training_smiles_lst)
  mu, cov = _calculate_distribution_statistics(chemnet, generated_smiles_lst)

  FCD = fcd.calculate_frechet_distance(mu1=mu_ref, mu2=mu,
                                     sigma1=cov_ref, sigma2=cov)
  score = np.exp(-0.2 * FCD)
  return score  

def fcd_distance_torch(generated_smiles_lst, training_smiles_lst):
  import os 
  os.environ['KMP_DUPLICATE_LIB_OK']='True'
  from fcd_torch import FCD
  fcd = FCD(device='cpu', n_jobs=8)
  return fcd(generated_smiles_lst, training_smiles_lst)

def fcd_distance(generated_smiles_lst, training_smiles_lst):
  try:
    import tensorflow, fcd
    global fcd 
  except:
    try:
      import torch, fcd_torch 
      return fcd_distance_torch(generated_smiles_lst, training_smiles_lst)
    except:
      raise ImportError("Please install fcd by 'pip install FCD' (for Tensorflow backend) \
                                            or 'pip install fcd_torch' (for PyTorch backend)!")
  return fcd_distance_tf(generated_smiles_lst, training_smiles_lst)



##################################################
#### End of distribution learning Evaluator
##################################################


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


fp2fpfunc = {'ECFP4': smiles_2_fingerprint_ECFP4, 
             'FCFP4': smiles_2_fingerprint_FCFP4, 
             'AP': smiles_2_fingerprint_AP, 
             'ECFP6': smiles_2_fingerprint_ECFP6
}

mean2func = {
  'geometric': gmean, 
  'arithmetic': np.mean, 
}


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
        if self.element == 'H':
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
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

    # Convert matches to the required format
    results = []
    for match in matches:
        # convert count to an integer, and set it to 1 if the count is not visible in the molecular formula
        count = 1 if not match[1] else int(match[1])
        results.append((match[0], count))

    return results
  
####################################################################
#################### isomer 
class Isomer_scoring:
  def __init__(self, target_smiles, means = 'geometric'):
    assert means in ['geometric', 'arithmetic']
    if means == 'geometric':
      self.mean_func = gmean 
    else: 
      self.mean_func = np.mean 
    atom2cnt_lst = parse_molecular_formula(target_smiles)
    total_atom_num = sum([cnt for atom,cnt in atom2cnt_lst]) 
    self.total_atom_modifier = GaussianModifier(mu=total_atom_num, sigma=2.0)
    self.AtomCounter_Modifier_lst = [((AtomCounter(atom)), GaussianModifier(mu=cnt,sigma=1.0)) for atom,cnt in atom2cnt_lst]

  def __call__(self, test_smiles):
    molecule = smiles_to_rdkit_mol(test_smiles)
    all_scores = []
    for atom_counter, modifier_func in self.AtomCounter_Modifier_lst:
      all_scores.append(modifier_func(atom_counter(molecule)))

    ### total atom number
    atom2cnt_lst = parse_molecular_formula(test_smiles)
    ## todo add Hs 
    total_atom_num = sum([cnt for atom,cnt in atom2cnt_lst])
    all_scores.append(self.total_atom_modifier(total_atom_num))
    return self.mean_func(all_scores)


def isomer_meta(target_smiles, means = 'geometric'):
  return Isomer_scoring(target_smiles, means = means)


isomers_c7h8n2o2 = isomer_meta(target_smiles = 'C7H8N2O2', means = 'geometric')
isomers_c9h10n2o2pf2cl = isomer_meta(target_smiles = 'C9H10N2O2PF2Cl', means = 'geometric')


####################################################################
#################### isomer 
####################################################################
#################### rediscovery 

class rediscovery_meta:
  def __init__(self, target_smiles, fp = 'ECFP4'):
    self.similarity_func = fp2fpfunc[fp]
    self.target_fp = self.similarity_func(target_smiles)

  def __call__(self, test_smiles):
    test_fp = self.similarity_func(test_smiles)
    similarity_value = DataStructs.TanimotoSimilarity(self.target_fp, test_fp)
    return similarity_value 


celecoxib_rediscovery = rediscovery_meta(target_smiles = 'CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F', fp = 'ECFP4')
troglitazone_rediscovery = rediscovery_meta(target_smiles = 'Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O', fp = 'ECFP4')
thiothixene_rediscovery = rediscovery_meta(target_smiles = 'CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1', fp = 'ECFP4')

####################################################################
#################### rediscovery 
####################################################################
#################### similarity 
class similarity_meta:
  def __init__(self, target_smiles, fp = 'FCFP4', modifier_func = None):
    self.similarity_func = fp2fpfunc[fp]
    self.target_fp = self.similarity_func(target_smiles)
    self.modifier_func = modifier_func 

  def __call__(self, test_smiles):
    test_fp = self.similarity_func(test_smiles)
    similarity_value = DataStructs.TanimotoSimilarity(self.target_fp, test_fp)
    if self.modifier_func is None:
      modifier_score = similarity_value
    else:
      modifier_score = self.modifier_func(similarity_value)
    return modifier_score 

similarity_modifier = ClippedScoreModifier(upper_x=0.75)
aripiprazole_similarity = similarity_meta(target_smiles = 'Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl', 
                                          fp = 'FCFP4', 
                                          modifier_func = similarity_modifier)

albuterol_similarity = similarity_meta(target_smiles = 'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1', 
                                       fp = 'FCFP4', 
                                       modifier_func = similarity_modifier)

mestranol_similarity = similarity_meta(target_smiles = 'COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1', 
                                       fp = 'AP', 
                                       modifier_func = similarity_modifier)

####################################################################
#################### similarity 
####################################################################
#################### median 

class median_meta:
  def __init__(self, target_smiles_1, target_smiles_2, fp1 = 'ECFP6', fp2 = 'ECFP6', modifier_func1 = None, modifier_func2 = None, means = 'geometric'):
    self.similarity_func1 = fp2fpfunc[fp1]
    self.similarity_func2 = fp2fpfunc[fp2]
    self.target_fp1 = self.similarity_func1(target_smiles_1)
    self.target_fp2 = self.similarity_func2(target_smiles_2)
    self.modifier_func1 = modifier_func1 
    self.modifier_func2 = modifier_func2 
    assert means in ['geometric', 'arithmetic']
    self.mean_func = mean2func[means]

  def __call__(self, test_smiles):
    test_fp1 = self.similarity_func1(test_smiles)
    test_fp2 = test_fp1 if self.similarity_func2 == self.similarity_func1 else self.similarity_func2(test_smiles)
    similarity_value1 = DataStructs.TanimotoSimilarity(self.target_fp1, test_fp1)
    similarity_value2 = DataStructs.TanimotoSimilarity(self.target_fp2, test_fp2)
    if self.modifier_func1 is None:
      modifier_score1 = similarity_value1
    else:
      modifier_score1 = self.modifier_func1(similarity_value1)
    if self.modifier_func2 is None:
      modifier_score2 = similarity_value2
    else:
      modifier_score2 = self.modifier_func2(similarity_value2)
    final_score = self.mean_func([modifier_score1 , modifier_score2])
    return final_score

camphor_smiles = 'CC1(C)C2CCC1(C)C(=O)C2'
menthol_smiles = 'CC(C)C1CCC(C)CC1O'

median1 = median_meta(target_smiles_1 = camphor_smiles, 
                      target_smiles_2 = menthol_smiles, 
                      fp1 = 'ECFP4', 
                      fp2 = 'ECFP4', 
                      modifier_func1 = None, 
                      modifier_func2 = None, 
                      means = 'geometric')

tadalafil_smiles = 'O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C'
sildenafil_smiles = 'CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C'
median2 = median_meta(target_smiles_1 = tadalafil_smiles, 
                      target_smiles_2 = sildenafil_smiles, 
                      fp1 = 'ECFP6', 
                      fp2 = 'ECFP6', 
                      modifier_func1 = None, 
                      modifier_func2 = None, 
                      means = 'geometric')

####################################################################
#################### median 
####################################################################
#################### MPO  

class MPO_meta:
  def __init__(self, means):
    '''
      target_smiles, fp in ['ECFP4', 'AP', ..., ]
      scoring, 
      modifier, 

    '''

    assert means in ['geometric', 'arithmetic']
    self.mean_func = mean2func[means]


  def __call__(self, test_smiles):
    molecule = smiles_to_rdkit_mol(test_smiles)

    score_lst = []
    return self.mean_func(score_lst)





def osimertinib_mpo(test_smiles):

  if 'osimertinib_fp_fcfc4' not in globals().keys():
    global osimertinib_fp_fcfc4, osimertinib_fp_ecfc6
    osimertinib_smiles = 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34'
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
  similarity_v1 = sim_v1_modifier(DataStructs.TanimotoSimilarity(osimertinib_fp_fcfc4, fp_fcfc4))
  similarity_v2 = sim_v2_modifier(DataStructs.TanimotoSimilarity(osimertinib_fp_ecfc6, fp_ecfc6))

  osimertinib_gmean = gmean([tpsa_score, logp_score, similarity_v1, similarity_v2])
  return osimertinib_gmean 





def fexofenadine_mpo(test_smiles):
  if 'fexofenadine_fp' not in globals().keys():
    global fexofenadine_fp
    fexofenadine_smiles = 'CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4'
    fexofenadine_fp = smiles_2_fingerprint_AP(fexofenadine_smiles)

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











def ranolazine_mpo(test_smiles):
  if 'ranolazine_fp' not in globals().keys():
    global ranolazine_fp, fluorine_counter  
    ranolazine_smiles = 'COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2'
    ranolazine_fp = smiles_2_fingerprint_AP(ranolazine_smiles)
    fluorine_counter = AtomCounter('F')

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










def perindopril_mpo(test_smiles):
  ## no similar_modifier

  if 'perindopril_fp' not in globals().keys():
    global perindopril_fp, num_aromatic_rings
    perindopril_smiles = 'O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC'
    perindopril_fp = smiles_2_fingerprint_ECFP4(perindopril_smiles)
    def num_aromatic_rings(mol):
      return rdMolDescriptors.CalcNumAromaticRings(mol)

  arom_rings_modifier = GaussianModifier(mu = 2, sigma = 0.5)

  molecule = smiles_to_rdkit_mol(test_smiles)
  fp_ecfp4 = smiles_2_fingerprint_ECFP4(test_smiles)

  similarity_value = DataStructs.TanimotoSimilarity(fp_ecfp4, perindopril_fp)
  num_aromatic_rings_value = arom_rings_modifier(num_aromatic_rings(molecule))

  perindopril_gmean = gmean([similarity_value, num_aromatic_rings_value])
  return perindopril_gmean










def amlodipine_mpo(test_smiles):
  ## no similar_modifier
  if 'amlodipine_fp' not in globals().keys():
    global amlodipine_fp, num_rings
    amlodipine_smiles = 'Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC'
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








def zaleplon_mpo(test_smiles):
  if 'zaleplon_fp' not in globals().keys():
    global zaleplon_fp, isomer_scoring_C19H17N3O2
    zaleplon_smiles = 'O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1'
    zaleplon_fp = smiles_2_fingerprint_ECFP4(zaleplon_smiles)
    isomer_scoring_C19H17N3O2 = Isomer_scoring(target_smiles = 'C19H17N3O2')

  fp = smiles_2_fingerprint_ECFP4(test_smiles)
  similarity_value = DataStructs.TanimotoSimilarity(fp, zaleplon_fp)
  isomer_value = isomer_scoring_C19H17N3O2(test_smiles)
  return gmean([similarity_value, isomer_value])

def sitagliptin_mpo(test_smiles):
  if 'sitagliptin_fp_ecfp4' not in globals().keys():
    global sitagliptin_fp_ecfp4, sitagliptin_logp_modifier, sitagliptin_tpsa_modifier, \
           isomers_scoring_C16H15F6N5O, sitagliptin_similar_modifier
    sitagliptin_smiles = 'Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F'
    sitagliptin_fp_ecfp4 = smiles_2_fingerprint_ECFP4(sitagliptin_smiles)
    sitagliptin_mol = Chem.MolFromSmiles(sitagliptin_smiles)
    sitagliptin_logp = Descriptors.MolLogP(sitagliptin_mol)
    sitagliptin_tpsa = Descriptors.TPSA(sitagliptin_mol)
    sitagliptin_logp_modifier = GaussianModifier(mu=sitagliptin_logp, sigma=0.2)
    sitagliptin_tpsa_modifier = GaussianModifier(mu=sitagliptin_tpsa, sigma=5)
    isomers_scoring_C16H15F6N5O = Isomer_scoring('C16H15F6N5O')
    sitagliptin_similar_modifier = GaussianModifier(mu=0, sigma=0.1)

  molecule = Chem.MolFromSmiles(test_smiles)
  fp_ecfp4 = smiles_2_fingerprint_ECFP4(test_smiles)
  logp_score = Descriptors.MolLogP(molecule)
  tpsa_score = Descriptors.TPSA(molecule)
  isomer_score = isomers_scoring_C16H15F6N5O(test_smiles)
  similarity_value = DataStructs.TanimotoSimilarity(fp_ecfp4, sitagliptin_fp_ecfp4)
  return gmean([similarity_value, logp_score, tpsa_score, isomer_score])







def get_PHCO_fingerprint(mol):
  if 'Gobbi_Pharm2D' not in globals().keys():
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
  if 'pharmacophor_fp' not in globals().keys():
    global pharmacophor_fp, deco1_smarts_scoring, deco2_smarts_scoring, scaffold_smarts_scoring   
    pharmacophor_smiles = 'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C'
    pharmacophor_mol = smiles_to_rdkit_mol(pharmacophor_smiles)
    pharmacophor_fp = get_PHCO_fingerprint(pharmacophor_mol)

    deco1_smarts_scoring = SMARTS_scoring(target_smarts = 'CS([#6])(=O)=O', inverse = True)
    deco2_smarts_scoring = SMARTS_scoring(target_smarts = '[#7]-c1ccc2ncsc2c1', inverse = True) 
    scaffold_smarts_scoring = SMARTS_scoring(target_smarts = '[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12', inverse = False) 

  molecule = smiles_to_rdkit_mol(test_smiles)
  fp = get_PHCO_fingerprint(molecule)
  similarity_modifier = ClippedScoreModifier(upper_x=0.85)

  similarity_value = similarity_modifier(DataStructs.TanimotoSimilarity(fp, pharmacophor_fp))
  deco1_score = deco1_smarts_scoring(molecule)
  deco2_score = deco2_smarts_scoring(molecule)
  scaffold_score = scaffold_smarts_scoring(molecule)

  all_scores = np.mean([similarity_value, deco1_score, deco2_score, scaffold_score])
  return all_scores







def scaffold_hop(test_smiles):
  if 'pharmacophor_fp' not in globals().keys() \
      or 'scaffold_smarts_scoring' not in globals().keys() \
      or 'deco_smarts_scoring' not in globals().keys():
    global pharmacophor_fp, deco_smarts_scoring, scaffold_smarts_scoring   
    pharmacophor_smiles = 'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C'
    pharmacophor_mol = smiles_to_rdkit_mol(pharmacophor_smiles)
    pharmacophor_fp = get_PHCO_fingerprint(pharmacophor_mol)

    deco_smarts_scoring = SMARTS_scoring(target_smarts = '[#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1', 
                                         inverse=False)

    scaffold_smarts_scoring = SMARTS_scoring(target_smarts = '[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12', 
                                             inverse=True)

  molecule = smiles_to_rdkit_mol(test_smiles)
  fp = get_PHCO_fingerprint(molecule)
  similarity_modifier = ClippedScoreModifier(upper_x=0.75)

  similarity_value = similarity_modifier(DataStructs.TanimotoSimilarity(fp, pharmacophor_fp))
  deco_score = deco_smarts_scoring(molecule)
  scaffold_score = scaffold_smarts_scoring(molecule)

  all_scores = np.mean([similarity_value, deco_score, scaffold_score])
  return all_scores








def valsartan_smarts(test_smiles):
  if 'valsartan_logp_modifier' not in globals().keys():
    global valsartan_mol, valsartan_logp_modifier, valsartan_tpsa_modifier, valsartan_bertz_modifier
    valsartan_smarts = 'CN(C=O)Cc1ccc(c2ccccc2)cc1' ### smarts 
    valsartan_mol = Chem.MolFromSmarts(valsartan_smarts)

    sitagliptin_smiles = 'NC(CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F' ### other mol
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
            resp = requests.get(host_ip + '/api/treebuilder/', params=params, verify=False)
            if 'error' not in resp.json().keys():
                break
                
    if save_json:
        with open(file_name, 'w') as f_data:
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
    """
    This function is modified from Dr. Jan Jensen's code
    """
    try:
      from rxn4chemistry import RXN4ChemistryWrapper
    except:
      print_sys("Please install rxn4chemistry via pip install rxn4chemistry")
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

# molecule.one

class molecule_one_retro:

    def __init__(self, api_token):
      try:
          from m1wrapper import MoleculeOneWrapper
      except:
          raise ImportError("Install Molecule.One Wrapper via pip install git+https://github.com/molecule-one/m1wrapper-python") 
      self.m1wrapper = MoleculeOneWrapper(api_token, 'https://app.molecule.one')
    

    def __call__(self, smiles):
      if isinstance(smiles, str):
          smiles = [smiles]

      search = self.m1wrapper.run_batch_search(
          targets=smiles,
          parameters={'exploratory_search': False, 'detail_level': 'score'}
      )

      status_cur = search.get_status()
      print_sys('Started Querying...')
      print_sys(status_cur)
      while True:
          time.sleep(7)
          status = search.get_status()

          if (status['queued'] == 0) and (status['running'] == 0):
              print_sys('Finished... Returning Results...')
              break
          else:
              if status_cur != status:
                  print_sys(status)
          status_cur = status
      result = search.get_results(precision=5, only=["targetSmiles", "result"])
      return {i['targetSmiles']: i['result'] for i in result}

class docking_meta:
    def __init__(self, software_calss='vina', pyscreener_path = './pyscreener', **kwargs):
        import sys
        sys.path.append(pyscreener_path)
        if software_calss == 'vina':
            from pyscreener.docking.vina import Vina as screener
        elif software_calss == 'dock6':
            from pyscreener.docking.dock import DOCK as screener
        else:
            raise ValueError("The value of software_calss is not implemented. Currently available:['vina', 'dock6']")

        self.scorer = screener(**kwargs)

    def __call__(self, test_smiles):
        final_score = self.scorer(test_smiles)
        return final_score

def smiles_to_rdkit_mol(smiles):
  mol = Chem.MolFromSmiles(smiles)
  #  Sanitization check (detects invalid valence)
  if mol is not None:
    try:
      Chem.SanitizeMol(mol)
    except ValueError:
      return None
  return mol


PubChemKeys = None


smartsPatts = {
1:('[H]', 3),# 1-115
2:('[H]', 7),
3:('[H]', 15),
4:('[H]', 31),
5:('[Li]', 0),
6:('[Li]', 1),
7:('[B]', 0),
8:('[B]', 1),
9:('[B]', 3),
10:('[C]', 1),
11:('[C]', 3),
12:('[C]', 7),
13:('[C]', 15),
14:('[C]', 31),
15:('[N]', 0),
16:('[N]', 1),
17:('[N]', 3),
18:('[N]', 7),
19:('[O]', 0),
20:('[O]', 1),
21:('[O]', 3),
22:('[O]', 7),
23:('[O]', 15),
24:('[F]', 0),
25:('[F]', 1),
26:('[F]', 3),
27:('[Na]', 0),
28:('[Na]', 1),
29:('[Si]', 0),
30:('[Si]', 1),
31:('[P]', 0),
32:('[P]', 1),
33:('[P]', 3),
34:('[S]', 0),
35:('[S]', 1),
36:('[S]', 3),
37:('[S]', 7),
38:('[Cl]', 0),
39:('[Cl]', 1),
40:('[Cl]', 3),
41:('[Cl]', 7),
42:('[K]', 0),
43:('[K]', 1),
44:('[Br]', 0),
45:('[Br]', 1),
46:('[Br]', 3),
47:('[I]', 0),
48:('[I]', 1),
49:('[I]', 3),
50:('[Be]', 0),
51:('[Mg]', 0),
52:('[Al]', 0),
53:('[Ca]', 0),
54:('[Sc]', 0),
55:('[Ti]', 0),
56:('[V]', 0),
57:('[Cr]', 0),
58:('[Mn]', 0),
59:('[Fe]', 0),
60:('[CO]', 0),
61:('[Ni]', 0),
62:('[Cu]', 0),
63:('[Zn]', 0),
64:('[Ga]', 0),
65:('[Ge]', 0),
66:('[As]', 0),
67:('[Se]', 0),
68:('[Kr]', 0),
69:('[Rb]', 0),
70:('[Sr]', 0),
71:('[Y]', 0),
72:('[Zr]', 0),
73:('[Nb]', 0),
74:('[Mo]', 0),
75:('[Ru]', 0),
76:('[Rh]', 0),
77:('[Pd]', 0),
78:('[Ag]', 0),
79:('[Cd]', 0),
80:('[In]', 0),
81:('[Sn]', 0),
82:('[Sb]', 0),
83:('[Te]', 0),
84:('[Xe]', 0),
85:('[Cs]', 0),
86:('[Ba]', 0),
87:('[Lu]', 0),
88:('[Hf]', 0),
89:('[Ta]', 0),
90:('[W]', 0),
91:('[Re]', 0),
92:('[Os]', 0),
93:('[Ir]', 0),
94:('[Pt]', 0),
95:('[Au]', 0),
96:('[Hg]', 0),
97:('[Tl]', 0),
98:('[Pb]', 0),
99:('[Bi]', 0),
100:('[La]', 0),
101:('[Ce]', 0),
102:('[Pr]', 0),
103:('[Nd]', 0),
104:('[Pm]', 0),
105:('[Sm]', 0),
106:('[Eu]', 0),
107:('[Gd]', 0),
108:('[Tb]', 0),
109:('[Dy]', 0),
110:('[Ho]', 0),
111:('[Er]', 0),
112:('[Tm]', 0),
113:('[Yb]', 0),
114:('[Tc]', 0),
115:('[U]', 0),
116:('[Li&!H0]', 0),#264-881
117:('[Li]~[Li]', 0),
118:('[Li]~[#5]', 0),
119:('[Li]~[#6]', 0),
120:('[Li]~[#8]', 0),
121:('[Li]~[F]', 0),
122:('[Li]~[#15]', 0),
123:('[Li]~[#16]', 0),
124:('[Li]~[Cl]', 0),
125:('[#5&!H0]', 0),
126:('[#5]~[#5]', 0),
127:('[#5]~[#6]', 0),
128:('[#5]~[#7]', 0),
129:('[#5]~[#8]', 0),
130:('[#5]~[F]', 0),
131:('[#5]~[#14]', 0),
132:('[#5]~[#15]', 0),
133:('[#5]~[#16]', 0),
134:('[#5]~[Cl]', 0),
135:('[#5]~[Br]', 0),
136:('[#6&!H0]', 0),
137:('[#6]~[#6]', 0),
138:('[#6]~[#7]', 0),
139:('[#6]~[#8]', 0),
140:('[#6]~[F]', 0),
141:('[#6]~[Na]', 0),
142:('[#6]~[Mg]', 0),
143:('[#6]~[Al]', 0),
144:('[#6]~[#14]', 0),
145:('[#6]~[#15]', 0),
146:('[#6]~[#16]', 0),
147:('[#6]~[Cl]', 0),
148:('[#6]~[#33]', 0),
149:('[#6]~[#34]', 0),
150:('[#6]~[Br]', 0),
151:('[#6]~[I]', 0),
152:('[#7&!H0]', 0),
153:('[#7]~[#7]', 0),
154:('[#7]~[#8]', 0),
155:('[#7]~[F]', 0),
156:('[#7]~[#14]', 0),
157:('[#7]~[#15]', 0),
158:('[#7]~[#16]', 0),
159:('[#7]~[Cl]', 0),
160:('[#7]~[Br]', 0),
161:('[#8&!H0]', 0),
162:('[#8]~[#8]', 0),
163:('[#8]~[Mg]', 0),
164:('[#8]~[Na]', 0),
165:('[#8]~[Al]', 0),
166:('[#8]~[#14]', 0),
167:('[#8]~[#15]', 0),
168:('[#8]~[K]', 0),
169:('[F]~[#15]', 0),
170:('[F]~[#16]', 0),
171:('[Al&!H0]', 0),
172:('[Al]~[Cl]', 0),
173:('[#14&!H0]', 0),
174:('[#14]~[#14]', 0),
175:('[#14]~[Cl]', 0),
176:('[#15&!H0]', 0),
177:('[#15]~[#15]', 0),
178:('[#33&!H0]', 0),
179:('[#33]~[#33]', 0),
180:('[#6](~Br)(~[#6])', 0),
181:('[#6](~Br)(~[#6])(~[#6])', 0),
182:('[#6&!H0]~[Br]', 0),
183:('[#6](~[Br])(:[c])', 0),
184:('[#6](~[Br])(:[n])', 0),
185:('[#6](~[#6])(~[#6])', 0),
186:('[#6](~[#6])(~[#6])(~[#6])', 0),
187:('[#6](~[#6])(~[#6])(~[#6])(~[#6])', 0),
188:('[#6H1](~[#6])(~[#6])(~[#6])', 0),
189:('[#6](~[#6])(~[#6])(~[#6])(~[#7])', 0),
190:('[#6](~[#6])(~[#6])(~[#6])(~[#8])', 0),
191:('[#6H1](~[#6])(~[#6])(~[#7])', 0),
192:('[#6H1](~[#6])(~[#6])(~[#8])', 0),
193:('[#6](~[#6])(~[#6])(~[#7])', 0),
194:('[#6](~[#6])(~[#6])(~[#8])', 0),
195:('[#6](~[#6])(~[Cl])', 0),
196:('[#6&!H0](~[#6])(~[Cl])', 0),
197:('[#6H,#6H2,#6H3,#6H4]~[#6]', 0),
198:('[#6&!H0](~[#6])(~[#7])', 0),
199:('[#6&!H0](~[#6])(~[#8])', 0),
200:('[#6H1](~[#6])(~[#8])(~[#8])', 0),
201:('[#6&!H0](~[#6])(~[#15])', 0),
202:('[#6&!H0](~[#6])(~[#16])', 0),
203:('[#6](~[#6])(~[I])', 0),
204:('[#6](~[#6])(~[#7])', 0),
205:('[#6](~[#6])(~[#8])', 0),
206:('[#6](~[#6])(~[#16])', 0),
207:('[#6](~[#6])(~[#14])', 0),
208:('[#6](~[#6])(:c)', 0),
209:('[#6](~[#6])(:c)(:c)', 0),
210:('[#6](~[#6])(:c)(:n)', 0),
211:('[#6](~[#6])(:n)', 0),
212:('[#6](~[#6])(:n)(:n)', 0),
213:('[#6](~[Cl])(~[Cl])', 0),
214:('[#6&!H0](~[Cl])', 0),
215:('[#6](~[Cl])(:c)', 0),
216:('[#6](~[F])(~[F])', 0),
217:('[#6](~[F])(:c)', 0),
218:('[#6&!H0](~[#7])', 0),
219:('[#6&!H0](~[#8])', 0),
220:('[#6&!H0](~[#8])(~[#8])', 0),
221:('[#6&!H0](~[#16])', 0),
222:('[#6&!H0](~[#14])', 0),
223:('[#6&!H0]:c', 0),
224:('[#6&!H0](:c)(:c)', 0),
225:('[#6&!H0](:c)(:n)', 0),
226:('[#6&!H0](:n)', 0),
227:('[#6H3]', 0),
228:('[#6](~[#7])(~[#7])', 0),
229:('[#6](~[#7])(:c)', 0),
230:('[#6](~[#7])(:c)(:c)', 0),
231:('[#6](~[#7])(:c)(:n)', 0),
232:('[#6](~[#7])(:n)', 0),
233:('[#6](~[#8])(~[#8])', 0),
234:('[#6](~[#8])(:c)', 0),
235:('[#6](~[#8])(:c)(:c)', 0),
236:('[#6](~[#16])(:c)', 0),
237:('[#6](:c)(:c)', 0),
238:('[#6](:c)(:c)(:c)', 0),
239:('[#6](:c)(:c)(:n)', 0),
240:('[#6](:c)(:n)', 0),
241:('[#6](:c)(:n)(:n)', 0),
242:('[#6](:n)(:n)', 0),
243:('[#7](~[#6])(~[#6])', 0),
244:('[#7](~[#6])(~[#6])(~[#6])', 0),
245:('[#7&!H0](~[#6])(~[#6])', 0),
246:('[#7&!H0](~[#6])', 0),
247:('[#7&!H0](~[#6])(~[#7])', 0),
248:('[#7](~[#6])(~[#8])', 0),
249:('[#7](~[#6])(:c)', 0),
250:('[#7](~[#6])(:c)(:c)', 0),
251:('[#7&!H0](~[#7])', 0),
252:('[#7&!H0](:c)', 0),
253:('[#7&!H0](:c)(:c)', 0),
254:('[#7](~[#8])(~[#8])', 0),
255:('[#7](~[#8])(:o)', 0),
256:('[#7](:c)(:c)', 0),
257:('[#7](:c)(:c)(:c)', 0),
258:('[#8](~[#6])(~[#6])', 0),
259:('[#8&!H0](~[#6])', 0),
260:('[#8](~[#6])(~[#15])', 0),
261:('[#8&!H0](~[#16])', 0),
262:('[#8](:c)(:c)', 0),
263:('[#15](~[#6])(~[#6])', 0),
264:('[#15](~[#8])(~[#8])', 0),
265:('[#16](~[#6])(~[#6])', 0),
266:('[#16&!H0](~[#6])', 0),
267:('[#16](~[#6])(~[#8])', 0),
268:('[#14](~[#6])(~[#6])', 0),
269:('[#6]=,:[#6]', 0),
270:('[#6]#[#6]', 0),
271:('[#6]=,:[#7]', 0),
272:('[#6]#[#7]', 0),
273:('[#6]=,:[#8]', 0),
274:('[#6]=,:[#16]', 0),
275:('[#7]=,:[#7]', 0),
276:('[#7]=,:[#8]', 0),
277:('[#7]=,:[#15]', 0),
278:('[#15]=,:[#8]', 0),
279:('[#15]=,:[#15]', 0),
280:('[#6](#[#6])(-,:[#6])', 0),
281:('[#6&!H0](#[#6])', 0),
282:('[#6](#[#7])(-,:[#6])', 0),
283:('[#6](-,:[#6])(-,:[#6])(=,:[#6])', 0),
284:('[#6](-,:[#6])(-,:[#6])(=,:[#7])', 0),
285:('[#6](-,:[#6])(-,:[#6])(=,:[#8])', 0),
286:('[#6](-,:[#6])([Cl])(=,:[#8])', 0),
287:('[#6&!H0](-,:[#6])(=,:[#6])', 0),
288:('[#6&!H0](-,:[#6])(=,:[#7])', 0),
289:('[#6&!H0](-,:[#6])(=,:[#8])', 0),
290:('[#6](-,:[#6])(-,:[#7])(=,:[#6])', 0),
291:('[#6](-,:[#6])(-,:[#7])(=,:[#7])', 0),
292:('[#6](-,:[#6])(-,:[#7])(=,:[#8])', 0),
293:('[#6](-,:[#6])(-,:[#8])(=,:[#8])', 0),
294:('[#6](-,:[#6])(=,:[#6])', 0),
295:('[#6](-,:[#6])(=,:[#7])', 0),
296:('[#6](-,:[#6])(=,:[#8])', 0),
297:('[#6]([Cl])(=,:[#8])', 0),
298:('[#6&!H0](-,:[#7])(=,:[#6])', 0),
299:('[#6&!H0](=,:[#6])', 0),
300:('[#6&!H0](=,:[#7])', 0),
301:('[#6&!H0](=,:[#8])', 0),
302:('[#6](-,:[#7])(=,:[#6])', 0),
303:('[#6](-,:[#7])(=,:[#7])', 0),
304:('[#6](-,:[#7])(=,:[#8])', 0),
305:('[#6](-,:[#8])(=,:[#8])', 0),
306:('[#7](-,:[#6])(=,:[#6])', 0),
307:('[#7](-,:[#6])(=,:[#8])', 0),
308:('[#7](-,:[#8])(=,:[#8])', 0),
309:('[#15](-,:[#8])(=,:[#8])', 0),
310:('[#16](-,:[#6])(=,:[#8])', 0),
311:('[#16](-,:[#8])(=,:[#8])', 0),
312:('[#16](=,:[#8])(=,:[#8])', 0),
313:('[#6]-,:[#6]-,:[#6]#[#6]', 0),
314:('[#8]-,:[#6]-,:[#6]=,:[#7]', 0),
315:('[#8]-,:[#6]-,:[#6]=,:[#8]', 0),
316:('[#7]:[#6]-,:[#16&!H0]', 0),
317:('[#7]-,:[#6]-,:[#6]=,:[#6]', 0),
318:('[#8]=,:[#16]-,:[#6]-,:[#6]', 0),
319:('[#7]#[#6]-,:[#6]=,:[#6]', 0),
320:('[#6]=,:[#7]-,:[#7]-,:[#6]', 0),
321:('[#8]=,:[#16]-,:[#6]-,:[#7]', 0),
322:('[#16]-,:[#16]-,:[#6]:[#6]', 0),
323:('[#6]:[#6]-,:[#6]=,:[#6]', 0),
324:('[#16]:[#6]:[#6]:[#6]', 0),
325:('[#6]:[#7]:[#6]-,:[#6]', 0),
326:('[#16]-,:[#6]:[#7]:[#6]', 0),
327:('[#16]:[#6]:[#6]:[#7]', 0),
328:('[#16]-,:[#6]=,:[#7]-,:[#6]', 0),
329:('[#6]-,:[#8]-,:[#6]=,:[#6]', 0),
330:('[#7]-,:[#7]-,:[#6]:[#6]', 0),
331:('[#16]-,:[#6]=,:[#7&!H0]', 0),
332:('[#16]-,:[#6]-,:[#16]-,:[#6]', 0),
333:('[#6]:[#16]:[#6]-,:[#6]', 0),
334:('[#8]-,:[#16]-,:[#6]:[#6]', 0),
335:('[#6]:[#7]-,:[#6]:[#6]', 0),
336:('[#7]-,:[#16]-,:[#6]:[#6]', 0),
337:('[#7]-,:[#6]:[#7]:[#6]', 0),
338:('[#7]:[#6]:[#6]:[#7]', 0),
339:('[#7]-,:[#6]:[#7]:[#7]', 0),
340:('[#7]-,:[#6]=,:[#7]-,:[#6]', 0),
341:('[#7]-,:[#6]=,:[#7&!H0]', 0),
342:('[#7]-,:[#6]-,:[#16]-,:[#6]', 0),
343:('[#6]-,:[#6]-,:[#6]=,:[#6]', 0),
344:('[#6]-,:[#7]:[#6&!H0]', 0),
345:('[#7]-,:[#6]:[#8]:[#6]', 0),
346:('[#8]=,:[#6]-,:[#6]:[#6]', 0),
347:('[#8]=,:[#6]-,:[#6]:[#7]', 0),
348:('[#6]-,:[#7]-,:[#6]:[#6]', 0),
349:('[#7]:[#7]-,:[#6&!H0]', 0),
350:('[#8]-,:[#6]:[#6]:[#7]', 0),
351:('[#8]-,:[#6]=,:[#6]-,:[#6]', 0),
352:('[#7]-,:[#6]:[#6]:[#7]', 0),
353:('[#6]-,:[#16]-,:[#6]:[#6]', 0),
354:('[Cl]-,:[#6]:[#6]-,:[#6]', 0),
355:('[#7]-,:[#6]=,:[#6&!H0]', 0),
356:('[Cl]-,:[#6]:[#6&!H0]', 0),
357:('[#7]:[#6]:[#7]-,:[#6]', 0),
358:('[Cl]-,:[#6]:[#6]-,:[#8]', 0),
359:('[#6]-,:[#6]:[#7]:[#6]', 0),
360:('[#6]-,:[#6]-,:[#16]-,:[#6]', 0),
361:('[#16]=,:[#6]-,:[#7]-,:[#6]', 0),
362:('[Br]-,:[#6]:[#6]-,:[#6]', 0),
363:('[#7&!H0]-,:[#7&!H0]', 0),
364:('[#16]=,:[#6]-,:[#7&!H0]', 0),
365:('[#6]-,:[#33]-[#8&!H0]', 0),
366:('[#16]:[#6]:[#6&!H0]', 0),
367:('[#8]-,:[#7]-,:[#6]-,:[#6]', 0),
368:('[#7]-,:[#7]-,:[#6]-,:[#6]', 0),
369:('[#6H,#6H2,#6H3]=,:[#6H,#6H2,#6H3]', 0),
370:('[#7]-,:[#7]-,:[#6]-,:[#7]', 0),
371:('[#8]=,:[#6]-,:[#7]-,:[#7]', 0),
372:('[#7]=,:[#6]-,:[#7]-,:[#6]', 0),
373:('[#6]=,:[#6]-,:[#6]:[#6]', 0),
374:('[#6]:[#7]-,:[#6&!H0]', 0),
375:('[#6]-,:[#7]-,:[#7&!H0]', 0),
376:('[#7]:[#6]:[#6]-,:[#6]', 0),
377:('[#6]-,:[#6]=,:[#6]-,:[#6]', 0),
378:('[#33]-,:[#6]:[#6&!H0]', 0),
379:('[Cl]-,:[#6]:[#6]-,:[Cl]', 0),
380:('[#6]:[#6]:[#7&!H0]', 0),
381:('[#7&!H0]-,:[#6&!H0]', 0),
382:('[Cl]-,:[#6]-,:[#6]-,:[Cl]', 0),
383:('[#7]:[#6]-,:[#6]:[#6]', 0),
384:('[#16]-,:[#6]:[#6]-,:[#6]', 0),
385:('[#16]-,:[#6]:[#6&!H0]', 0),
386:('[#16]-,:[#6]:[#6]-,:[#7]', 0),
387:('[#16]-,:[#6]:[#6]-,:[#8]', 0),
388:('[#8]=,:[#6]-,:[#6]-,:[#6]', 0),
389:('[#8]=,:[#6]-,:[#6]-,:[#7]', 0),
390:('[#8]=,:[#6]-,:[#6]-,:[#8]', 0),
391:('[#7]=,:[#6]-,:[#6]-,:[#6]', 0),
392:('[#7]=,:[#6]-,:[#6&!H0]', 0),
393:('[#6]-,:[#7]-,:[#6&!H0]', 0),
394:('[#8]-,:[#6]:[#6]-,:[#6]', 0),
395:('[#8]-,:[#6]:[#6&!H0]', 0),
396:('[#8]-,:[#6]:[#6]-,:[#7]', 0),
397:('[#8]-,:[#6]:[#6]-,:[#8]', 0),
398:('[#7]-,:[#6]:[#6]-,:[#6]', 0),
399:('[#7]-,:[#6]:[#6&!H0]', 0),
400:('[#7]-,:[#6]:[#6]-,:[#7]', 0),
401:('[#8]-,:[#6]-,:[#6]:[#6]', 0),
402:('[#7]-,:[#6]-,:[#6]:[#6]', 0),
403:('[Cl]-,:[#6]-,:[#6]-,:[#6]', 0),
404:('[Cl]-,:[#6]-,:[#6]-,:[#8]', 0),
405:('[#6]:[#6]-,:[#6]:[#6]', 0),
406:('[#8]=,:[#6]-,:[#6]=,:[#6]', 0),
407:('[Br]-,:[#6]-,:[#6]-,:[#6]', 0),
408:('[#7]=,:[#6]-,:[#6]=,:[#6]', 0),
409:('[#6]=,:[#6]-,:[#6]-,:[#6]', 0),
410:('[#7]:[#6]-,:[#8&!H0]', 0),
411:('[#8]=,:[#7]-,:c:c', 0),
412:('[#8]-,:[#6]-,:[#7&!H0]', 0),
413:('[#7]-,:[#6]-,:[#7]-,:[#6]', 0),
414:('[Cl]-,:[#6]-,:[#6]=,:[#8]', 0),
415:('[Br]-,:[#6]-,:[#6]=,:[#8]', 0),
416:('[#8]-,:[#6]-,:[#8]-,:[#6]', 0),
417:('[#6]=,:[#6]-,:[#6]=,:[#6]', 0),
418:('[#6]:[#6]-,:[#8]-,:[#6]', 0),
419:('[#8]-,:[#6]-,:[#6]-,:[#7]', 0),
420:('[#8]-,:[#6]-,:[#6]-,:[#8]', 0),
421:('N#[#6]-,:[#6]-,:[#6]', 0),
422:('[#7]-,:[#6]-,:[#6]-,:[#7]', 0),
423:('[#6]:[#6]-,:[#6]-,:[#6]', 0),
424:('[#6&!H0]-,:[#8&!H0]', 0),
425:('n:c:n:c', 0),
426:('[#8]-,:[#6]-,:[#6]=,:[#6]', 0),
427:('[#8]-,:[#6]-,:[#6]:[#6]-,:[#6]', 0),
428:('[#8]-,:[#6]-,:[#6]:[#6]-,:[#8]', 0),
429:('[#7]=,:[#6]-,:[#6]:[#6&!H0]', 0),
430:('c:c-,:[#7]-,:c:c', 0),
431:('[#6]-,:[#6]:[#6]-,:c:c', 0),
432:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
433:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
434:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
435:('[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
436:('[Cl]-,:[#6]:[#6]-,:[#8]-,:[#6]', 0),
437:('c:c-,:[#6]=,:[#6]-,:[#6]', 0),
438:('[#6]-,:[#6]:[#6]-,:[#7]-,:[#6]', 0),
439:('[#6]-,:[#16]-,:[#6]-,:[#6]-,:[#6]', 0),
440:('[#7]-,:[#6]:[#6]-,:[#8&!H0]', 0),
441:('[#8]=,:[#6]-,:[#6]-,:[#6]=,:[#8]', 0),
442:('[#6]-,:[#6]:[#6]-,:[#8]-,:[#6]', 0),
443:('[#6]-,:[#6]:[#6]-,:[#8&!H0]', 0),
444:('[Cl]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
445:('[#7]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
446:('[#7]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
447:('[#6]-,:[#8]-,:[#6]-,:[#6]=,:[#6]', 0),
448:('c:c-,:[#6]-,:[#6]-,:[#6]', 0),
449:('[#7]=,:[#6]-,:[#7]-,:[#6]-,:[#6]', 0),
450:('[#8]=,:[#6]-,:[#6]-,:c:c', 0),
451:('[Cl]-,:[#6]:[#6]:[#6]-,:[#6]', 0),
452:('[#6H,#6H2,#6H3]-,:[#6]=,:[#6H,#6H2,#6H3]', 0),
453:('[#7]-,:[#6]:[#6]:[#6]-,:[#6]', 0),
454:('[#7]-,:[#6]:[#6]:[#6]-,:[#7]', 0),
455:('[#8]=,:[#6]-,:[#6]-,:[#7]-,:[#6]', 0),
456:('[#6]-,:c:c:[#6]-,:[#6]', 0),
457:('[#6]-,:[#8]-,:[#6]-,:[#6]:c', 0),
458:('[#8]=,:[#6]-,:[#6]-,:[#8]-,:[#6]', 0),
459:('[#8]-,:[#6]:[#6]-,:[#6]-,:[#6]', 0),
460:('[#7]-,:[#6]-,:[#6]-,:[#6]:c', 0),
461:('[#6]-,:[#6]-,:[#6]-,:[#6]:c', 0),
462:('[Cl]-,:[#6]-,:[#6]-,:[#7]-,:[#6]', 0),
463:('[#6]-,:[#8]-,:[#6]-,:[#8]-,:[#6]', 0),
464:('[#7]-,:[#6]-,:[#6]-,:[#7]-,:[#6]', 0),
465:('[#7]-,:[#6]-,:[#8]-,:[#6]-,:[#6]', 0),
466:('[#6]-,:[#7]-,:[#6]-,:[#6]-,:[#6]', 0),
467:('[#6]-,:[#6]-,:[#8]-,:[#6]-,:[#6]', 0),
468:('[#7]-,:[#6]-,:[#6]-,:[#8]-,:[#6]', 0),
469:('c:c:n:n:c', 0),
470:('[#6]-,:[#6]-,:[#6]-,:[#8&!H0]', 0),
471:('c:[#6]-,:[#6]-,:[#6]:c', 0),
472:('[#8]-,:[#6]-,:[#6]=,:[#6]-,:[#6]', 0),
473:('c:c-,:[#8]-,:[#6]-,:[#6]', 0),
474:('[#7]-,:[#6]:c:c:n', 0),
475:('[#8]=,:[#6]-,:[#8]-,:[#6]:c', 0),
476:('[#8]=,:[#6]-,:[#6]:[#6]-,:[#6]', 0),
477:('[#8]=,:[#6]-,:[#6]:[#6]-,:[#7]', 0),
478:('[#8]=,:[#6]-,:[#6]:[#6]-,:[#8]', 0),
479:('[#6]-,:[#8]-,:[#6]:[#6]-,:[#6]', 0),
480:('[#8]=,:[#33]-,:[#6]:c:c', 0),
481:('[#6]-,:[#7]-,:[#6]-,:[#6]:c', 0),
482:('[#16]-,:[#6]:c:c-,:[#7]', 0),
483:('[#8]-,:[#6]:[#6]-,:[#8]-,:[#6]', 0),
484:('[#8]-,:[#6]:[#6]-,:[#8&!H0]', 0),
485:('[#6]-,:[#6]-,:[#8]-,:[#6]:c', 0),
486:('[#7]-,:[#6]-,:[#6]:[#6]-,:[#6]', 0),
487:('[#6]-,:[#6]-,:[#6]:[#6]-,:[#6]', 0),
488:('[#7]-,:[#7]-,:[#6]-,:[#7&!H0]', 0),
489:('[#6]-,:[#7]-,:[#6]-,:[#7]-,:[#6]', 0),
490:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
491:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
492:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
493:('[#6]=,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
494:('[#8]-,:[#6]-,:[#6]-,:[#6]=,:[#6]', 0),
495:('[#8]-,:[#6]-,:[#6]-,:[#6]=,:[#8]', 0),
496:('[#6&!H0]-,:[#6]-,:[#7&!H0]', 0),
497:('[#6]-,:[#6]=,:[#7]-,:[#7]-,:[#6]', 0),
498:('[#8]=,:[#6]-,:[#7]-,:[#6]-,:[#6]', 0),
499:('[#8]=,:[#6]-,:[#7]-,:[#6&!H0]', 0),
500:('[#8]=,:[#6]-,:[#7]-,:[#6]-,:[#7]', 0),
501:('[#8]=,:[#7]-,:[#6]:[#6]-,:[#7]', 0),
502:('[#8]=,:[#7]-,:c:c-,:[#8]', 0),
503:('[#8]=,:[#6]-,:[#7]-,:[#6]=,:[#8]', 0),
504:('[#8]-,:[#6]:[#6]:[#6]-,:[#6]', 0),
505:('[#8]-,:[#6]:[#6]:[#6]-,:[#7]', 0),
506:('[#8]-,:[#6]:[#6]:[#6]-,:[#8]', 0),
507:('[#7]-,:[#6]-,:[#7]-,:[#6]-,:[#6]', 0),
508:('[#8]-,:[#6]-,:[#6]-,:[#6]:c', 0),
509:('[#6]-,:[#6]-,:[#7]-,:[#6]-,:[#6]', 0),
510:('[#6]-,:[#7]-,:[#6]:[#6]-,:[#6]', 0),
511:('[#6]-,:[#6]-,:[#16]-,:[#6]-,:[#6]', 0),
512:('[#8]-,:[#6]-,:[#6]-,:[#7]-,:[#6]', 0),
513:('[#6]-,:[#6]=,:[#6]-,:[#6]-,:[#6]', 0),
514:('[#8]-,:[#6]-,:[#8]-,:[#6]-,:[#6]', 0),
515:('[#8]-,:[#6]-,:[#6]-,:[#8]-,:[#6]', 0),
516:('[#8]-,:[#6]-,:[#6]-,:[#8&!H0]', 0),
517:('[#6]-,:[#6]=,:[#6]-,:[#6]=,:[#6]', 0),
518:('[#7]-,:[#6]:[#6]-,:[#6]-,:[#6]', 0),
519:('[#6]=,:[#6]-,:[#6]-,:[#8]-,:[#6]', 0),
520:('[#6]=,:[#6]-,:[#6]-,:[#8&!H0]', 0),
521:('[#6]-,:[#6]:[#6]-,:[#6]-,:[#6]', 0),
522:('[Cl]-,:[#6]:[#6]-,:[#6]=,:[#8]', 0),
523:('[Br]-,:[#6]:c:c-,:[#6]', 0),
524:('[#8]=,:[#6]-,:[#6]=,:[#6]-,:[#6]', 0),
525:('[#8]=,:[#6]-,:[#6]=,:[#6&!H0]', 0),
526:('[#8]=,:[#6]-,:[#6]=,:[#6]-,:[#7]', 0),
527:('[#7]-,:[#6]-,:[#7]-,:[#6]:c', 0),
528:('[Br]-,:[#6]-,:[#6]-,:[#6]:c', 0),
529:('[#7]#[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
530:('[#6]-,:[#6]=,:[#6]-,:[#6]:c', 0),
531:('[#6]-,:[#6]-,:[#6]=,:[#6]-,:[#6]', 0),
532:('[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
533:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
534:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
535:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
536:('[#7]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
537:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
538:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
539:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
540:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]=,:[#8]', 0),
541:('[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
542:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
543:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
544:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
545:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
546:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]', 0),
547:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]=,:[#8]', 0),
548:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]', 0),
549:('[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
550:('[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#6])-,:[#6]', 0),
551:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
552:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#6])-,:[#6]', 0),
553:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#8]-,:[#6]', 0),
554:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#8])-,:[#6]', 0),
555:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#7]-,:[#6]', 0),
556:('[#8]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#7])-,:[#6]', 0),
557:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]', 0),
558:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#8])-,:[#6]', 0),
559:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](=,:[#8])-,:[#6]', 0),
560:('[#8]=,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6](-,:[#7])-,:[#6]', 0),
561:('[#6]-,:[#6](-,:[#6])-,:[#6]-,:[#6]', 0),
562:('[#6]-,:[#6](-,:[#6])-,:[#6]-,:[#6]-,:[#6]', 0),
563:('[#6]-,:[#6]-,:[#6](-,:[#6])-,:[#6]-,:[#6]', 0),
564:('[#6]-,:[#6](-,:[#6])(-,:[#6])-,:[#6]-,:[#6]', 0),
565:('[#6]-,:[#6](-,:[#6])-,:[#6](-,:[#6])-,:[#6]', 0),
566:('[#6]c1ccc([#6])cc1', 0),
567:('[#6]c1ccc([#8])cc1', 0),
568:('[#6]c1ccc([#16])cc1', 0),
569:('[#6]c1ccc([#7])cc1', 0),
570:('[#6]c1ccc(Cl)cc1', 0),
571:('[#6]c1ccc(Br)cc1', 0),
572:('[#8]c1ccc([#8])cc1', 0),
573:('[#8]c1ccc([#16])cc1', 0),
574:('[#8]c1ccc([#7])cc1', 0),
575:('[#8]c1ccc(Cl)cc1', 0),
576:('[#8]c1ccc(Br)cc1', 0),
577:('[#16]c1ccc([#16])cc1', 0),
578:('[#16]c1ccc([#7])cc1', 0),
579:('[#16]c1ccc(Cl)cc1', 0),
580:('[#16]c1ccc(Br)cc1', 0),
581:('[#7]c1ccc([#7])cc1', 0),
582:('[#7]c1ccc(Cl)cc1', 0),
583:('[#7]c1ccc(Br)cc1', 0),
584:('Clc1ccc(Cl)cc1', 0),
585:('Clc1ccc(Br)cc1', 0),
586:('Brc1ccc(Br)cc1', 0),
587:('[#6]c1cc([#6])ccc1', 0),
588:('[#6]c1cc([#8])ccc1', 0),
589:('[#6]c1cc([#16])ccc1', 0),
590:('[#6]c1cc([#7])ccc1', 0),
591:('[#6]c1cc(Cl)ccc1', 0),
592:('[#6]c1cc(Br)ccc1', 0),
593:('[#8]c1cc([#8])ccc1', 0),
594:('[#8]c1cc([#16])ccc1', 0),
595:('[#8]c1cc([#7])ccc1', 0),
596:('[#8]c1cc(Cl)ccc1', 0),
597:('[#8]c1cc(Br)ccc1', 0),
598:('[#16]c1cc([#16])ccc1', 0),
599:('[#16]c1cc([#7])ccc1', 0),
600:('[#16]c1cc(Cl)ccc1', 0),
601:('[#16]c1cc(Br)ccc1', 0),
602:('[#7]c1cc([#7])ccc1', 0),
603:('[#7]c1cc(Cl)ccc1', 0),
604:('[#7]c1cc(Br)ccc1', 0),
605:('Clc1cc(Cl)ccc1', 0),
606:('Clc1cc(Br)ccc1', 0),
607:('Brc1cc(Br)ccc1', 0),
608:('[#6]c1c([#6])cccc1', 0),
609:('[#6]c1c([#8])cccc1', 0),
610:('[#6]c1c([#16])cccc1', 0),
611:('[#6]c1c([#7])cccc1', 0),
612:('[#6]c1c(Cl)cccc1', 0),
613:('[#6]c1c(Br)cccc1', 0),
614:('[#8]c1c([#8])cccc1', 0),
615:('[#8]c1c([#16])cccc1', 0),
616:('[#8]c1c([#7])cccc1', 0),
617:('[#8]c1c(Cl)cccc1', 0),
618:('[#8]c1c(Br)cccc1', 0),
619:('[#16]c1c([#16])cccc1', 0),
620:('[#16]c1c([#7])cccc1', 0),
621:('[#16]c1c(Cl)cccc1', 0),
622:('[#16]c1c(Br)cccc1', 0),
623:('[#7]c1c([#7])cccc1', 0),
624:('[#7]c1c(Cl)cccc1', 0),
625:('[#7]c1c(Br)cccc1', 0),
626:('Clc1c(Cl)cccc1', 0),
627:('Clc1c(Br)cccc1', 0),
628:('Brc1c(Br)cccc1', 0),
629:('[#6][#6]1[#6][#6][#6]([#6])[#6][#6]1', 0),
630:('[#6][#6]1[#6][#6][#6]([#8])[#6][#6]1', 0),
631:('[#6][#6]1[#6][#6][#6]([#16])[#6][#6]1', 0),
632:('[#6][#6]1[#6][#6][#6]([#7])[#6][#6]1', 0),
633:('[#6][#6]1[#6][#6][#6](Cl)[#6][#6]1', 0),
634:('[#6][#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
635:('[#8][#6]1[#6][#6][#6]([#8])[#6][#6]1', 0),
636:('[#8][#6]1[#6][#6][#6]([#16])[#6][#6]1', 0),
637:('[#8][#6]1[#6][#6][#6]([#7])[#6][#6]1', 0),
638:('[#8][#6]1[#6][#6][#6](Cl)[#6][#6]1', 0),
639:('[#8][#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
640:('[#16][#6]1[#6][#6][#6]([#16])[#6][#6]1', 0),
641:('[#16][#6]1[#6][#6][#6]([#7])[#6][#6]1', 0),
642:('[#16][#6]1[#6][#6][#6](Cl)[#6][#6]1', 0),
643:('[#16][#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
644:('[#7][#6]1[#6][#6][#6]([#7])[#6][#6]1', 0),
645:('[#7][#6]1[#6][#6][#6](Cl)[#6][#6]1', 0),
646:('[#7][#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
647:('Cl[#6]1[#6][#6][#6](Cl)[#6][#6]1', 0),
648:('Cl[#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
649:('Br[#6]1[#6][#6][#6](Br)[#6][#6]1', 0),
650:('[#6][#6]1[#6][#6]([#6])[#6][#6][#6]1', 0),
651:('[#6][#6]1[#6][#6]([#8])[#6][#6][#6]1', 0),
652:('[#6][#6]1[#6][#6]([#16])[#6][#6][#6]1', 0),
653:('[#6][#6]1[#6][#6]([#7])[#6][#6][#6]1', 0),
654:('[#6][#6]1[#6][#6](Cl)[#6][#6][#6]1', 0),
655:('[#6][#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
656:('[#8][#6]1[#6][#6]([#8])[#6][#6][#6]1', 0),
657:('[#8][#6]1[#6][#6]([#16])[#6][#6][#6]1', 0),
658:('[#8][#6]1[#6][#6]([#7])[#6][#6][#6]1', 0),
659:('[#8][#6]1[#6][#6](Cl)[#6][#6][#6]1', 0),
660:('[#8][#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
661:('[#16][#6]1[#6][#6]([#16])[#6][#6][#6]1', 0),
662:('[#16][#6]1[#6][#6]([#7])[#6][#6][#6]1', 0),
663:('[#16][#6]1[#6][#6](Cl)[#6][#6][#6]1', 0),
664:('[#16][#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
665:('[#7][#6]1[#6][#6]([#7])[#6][#6][#6]1', 0),
666:('[#7][#6]1[#6][#6](Cl)[#6][#6][#6]1', 0),
667:('[#7][#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
668:('Cl[#6]1[#6][#6](Cl)[#6][#6][#6]1', 0),
669:('Cl[#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
670:('Br[#6]1[#6][#6](Br)[#6][#6][#6]1', 0),
671:('[#6][#6]1[#6]([#6])[#6][#6][#6][#6]1', 0),
672:('[#6][#6]1[#6]([#8])[#6][#6][#6][#6]1', 0),
673:('[#6][#6]1[#6]([#16])[#6][#6][#6][#6]1', 0),
674:('[#6][#6]1[#6]([#7])[#6][#6][#6][#6]1', 0),
675:('[#6][#6]1[#6](Cl)[#6][#6][#6][#6]1', 0),
676:('[#6][#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
677:('[#8][#6]1[#6]([#8])[#6][#6][#6][#6]1', 0),
678:('[#8][#6]1[#6]([#16])[#6][#6][#6][#6]1', 0),
679:('[#8][#6]1[#6]([#7])[#6][#6][#6][#6]1', 0),
680:('[#8][#6]1[#6](Cl)[#6][#6][#6][#6]1', 0),
681:('[#8][#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
682:('[#16][#6]1[#6]([#16])[#6][#6][#6][#6]1', 0),
683:('[#16][#6]1[#6]([#7])[#6][#6][#6][#6]1', 0),
684:('[#16][#6]1[#6](Cl)[#6][#6][#6][#6]1', 0),
685:('[#16][#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
686:('[#7][#6]1[#6]([#7])[#6][#6][#6][#6]1', 0),
687:('[#7][#6]1[#6](Cl)[#6][#6][#6][#6]1', 0),
688:('[#7][#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
689:('Cl[#6]1[#6](Cl)[#6][#6][#6][#6]1', 0),
690:('Cl[#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
691:('Br[#6]1[#6](Br)[#6][#6][#6][#6]1', 0),
692:('[#6][#6]1[#6][#6]([#6])[#6][#6]1', 0),
693:('[#6][#6]1[#6][#6]([#8])[#6][#6]1', 0),
694:('[#6][#6]1[#6][#6]([#16])[#6][#6]1', 0),
695:('[#6][#6]1[#6][#6]([#7])[#6][#6]1', 0),
696:('[#6][#6]1[#6][#6](Cl)[#6][#6]1', 0),
697:('[#6][#6]1[#6][#6](Br)[#6][#6]1', 0),
698:('[#8][#6]1[#6][#6]([#8])[#6][#6]1', 0),
699:('[#8][#6]1[#6][#6]([#16])[#6][#6]1', 0),
700:('[#8][#6]1[#6][#6]([#7])[#6][#6]1', 0),
701:('[#8][#6]1[#6][#6](Cl)[#6][#6]1', 0),
702:('[#8][#6]1[#6][#6](Br)[#6][#6]1', 0),
703:('[#16][#6]1[#6][#6]([#16])[#6][#6]1', 0),
704:('[#16][#6]1[#6][#6]([#7])[#6][#6]1', 0),
705:('[#16][#6]1[#6][#6](Cl)[#6][#6]1', 0),
706:('[#16][#6]1[#6][#6](Br)[#6][#6]1', 0),
707:('[#7][#6]1[#6][#6]([#7])[#6][#6]1', 0),
708:('[#7][#6]1[#6][#6](Cl)[#6][#6]1', 0),
709:('[#7][#6]1[#6][#6](Br)[#6][#6]1', 0),
710:('Cl[#6]1[#6][#6](Cl)[#6][#6]1', 0),
711:('Cl[#6]1[#6][#6](Br)[#6][#6]1', 0),
712:('Br[#6]1[#6][#6](Br)[#6][#6]1', 0),
713:('[#6][#6]1[#6]([#6])[#6][#6][#6]1', 0),
714:('[#6][#6]1[#6]([#8])[#6][#6][#6]1', 0),
715:('[#6][#6]1[#6]([#16])[#6][#6][#6]1', 0),
716:('[#6][#6]1[#6]([#7])[#6][#6][#6]1', 0),
717:('[#6][#6]1[#6](Cl)[#6][#6][#6]1', 0),
718:('[#6][#6]1[#6](Br)[#6][#6][#6]1', 0),
719:('[#8][#6]1[#6]([#8])[#6][#6][#6]1', 0),
720:('[#8][#6]1[#6]([#16])[#6][#6][#6]1', 0),
721:('[#8][#6]1[#6]([#7])[#6][#6][#6]1', 0),
722:('[#8][#6]1[#6](Cl)[#6][#6][#6]1', 0),
723:('[#8][#6]1[#6](Br)[#6][#6][#6]1', 0),
724:('[#16][#6]1[#6]([#16])[#6][#6][#6]1', 0),
725:('[#16][#6]1[#6]([#7])[#6][#6][#6]1', 0),
726:('[#16][#6]1[#6](Cl)[#6][#6][#6]1', 0),
727:('[#16][#6]1[#6](Br)[#6][#6][#6]1', 0),
728:('[#7][#6]1[#6]([#7])[#6][#6][#6]1', 0),
729:('[#7][#6]1[#6](Cl)[#6][#6]1', 0),
730:('[#7][#6]1[#6](Br)[#6][#6][#6]1', 0),
731:('Cl[#6]1[#6](Cl)[#6][#6][#6]1', 0),
732:('Cl[#6]1[#6](Br)[#6][#6][#6]1', 0),
733:('Br[#6]1[#6](Br)[#6][#6][#6]1', 0)}


def InitKeys(keyList, keyDict):
  assert len(keyList) == len(keyDict.keys()), 'length mismatch'
  for key in keyDict.keys():
    patt, count = keyDict[key]
    if patt != '?':
      sma = Chem.MolFromSmarts(patt)
      if not sma:
        print_sys('SMARTS parser error for key #%d: %s' % (key, patt))
      else:
        keyList[key - 1] = sma, count


def calcPubChemFingerPart1(mol, **kwargs):
  global PubChemKeys
  if PubChemKeys is None:
    PubChemKeys = [(None, 0)] * len(smartsPatts.keys())

    InitKeys(PubChemKeys, smartsPatts)
  ctor = kwargs.get('ctor', DataStructs.SparseBitVect)

  res = ctor(len(PubChemKeys) + 1)
  for i, (patt, count) in enumerate(PubChemKeys):
    if patt is not None:
      if count == 0:
        res[i + 1] = mol.HasSubstructMatch(patt)
      else:
        matches = mol.GetSubstructMatches(patt)
        if len(matches) > count:
          res[i + 1] = 1
  return res


def func_1(mol,bits):
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    AllRingsAtom = mol.GetRingInfo().AtomRings()
    for ring in AllRingsAtom:
        ringSize.append(len(ring))
        for k,v in temp.items():
            if len(ring) == k:
                temp[k]+=1
    if temp[3]>=2:
        bits[0]=1;bits[7]=1
    elif temp[3]==1:
        bits[0]=1
    else:
        pass
    if temp[4]>=2:
        bits[14]=1;bits[21]=1
    elif temp[4]==1:
        bits[14]=1
    else:
        pass
    if temp[5]>=5:
        bits[28]=1;bits[35]=1;bits[42]=1;bits[49]=1;bits[56]=1
    elif temp[5]==4:
        bits[28]=1;bits[35]=1;bits[42]=1;bits[49]=1
    elif temp[5]==3:
        bits[28]=1;bits[35]=1;bits[42]=1
    elif temp[5]==2:
        bits[28]=1;bits[35]=1
    elif temp[5]==1:
        bits[28]=1
    else:
        pass
    if temp[6]>=5:
        bits[63]=1;bits[70]=1;bits[77]=1;bits[84]=1;bits[91]=1
    elif temp[6]==4:
        bits[63]=1;bits[70]=1;bits[77]=1;bits[84]=1
    elif temp[6]==3:
        bits[63]=1;bits[70]=1;bits[77]=1
    elif temp[6]==2:
        bits[63]=1;bits[70]=1
    elif temp[6]==1:
        bits[63]=1
    else:
        pass
    if temp[7]>=2:
        bits[98]=1;bits[105]=1
    elif temp[7]==1:
        bits[98]=1
    else:
        pass
    if temp[8]>=2:
        bits[112]=1;bits[119]=1
    elif temp[8]==1:
        bits[112]=1
    else:
        pass
    if temp[9]>=1:
        bits[126]=1;
    else:
        pass
    if temp[10]>=1:
        bits[133]=1;
    else:
        pass

    return ringSize,bits


def func_2(mol,bits):
    """ *Internal Use Only*
    saturated or aromatic carbon-only ring
    """
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
        ######## aromatic carbon-only     
        aromatic = True
        AllCarb = True
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() != 6 or EndAtom.GetAtomicNum() != 6:
                AllCarb = False
                break
        if aromatic == True and AllCarb == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[1]=1;bits[8]=1
    elif temp[3]==1:
        bits[1]=1
    else:
        pass
    if temp[4]>=2:
        bits[15]=1;bits[22]=1
    elif temp[4]==1:
        bits[15]=1
    else:
        pass
    if temp[5]>=5:
        bits[29]=1;bits[36]=1;bits[43]=1;bits[50]=1;bits[57]=1
    elif temp[5]==4:
        bits[29]=1;bits[36]=1;bits[43]=1;bits[50]=1
    elif temp[5]==3:
        bits[29]=1;bits[36]=1;bits[43]=1
    elif temp[5]==2:
        bits[29]=1;bits[36]=1
    elif temp[5]==1:
        bits[29]=1
    else:
        pass
    if temp[6]>=5:
        bits[64]=1;bits[71]=1;bits[78]=1;bits[85]=1;bits[92]=1
    elif temp[6]==4:
        bits[64]=1;bits[71]=1;bits[78]=1;bits[85]=1
    elif temp[6]==3:
        bits[64]=1;bits[71]=1;bits[78]=1
    elif temp[6]==2:
        bits[64]=1;bits[71]=1
    elif temp[6]==1:
        bits[64]=1
    else:
        pass
    if temp[7]>=2:
        bits[99]=1;bits[106]=1
    elif temp[7]==1:
        bits[99]=1
    else:
        pass
    if temp[8]>=2:
        bits[113]=1;bits[120]=1
    elif temp[8]==1:
        bits[113]=1
    else:
        pass
    if temp[9]>=1:
        bits[127]=1;
    else:
        pass
    if temp[10]>=1:
        bits[134]=1;
    else:
        pass
    return ringSize, bits


def func_3(mol,bits):
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
        ######## aromatic nitrogen-containing    
        aromatic = True
        ContainNitro = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() == 7 or EndAtom.GetAtomicNum() == 7:
                ContainNitro = True
                break
        if aromatic == True and ContainNitro == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[2]=1;bits[9]=1
    elif temp[3]==1:
        bits[2]=1
    else:
        pass
    if temp[4]>=2:
        bits[16]=1;bits[23]=1
    elif temp[4]==1:
        bits[16]=1
    else:
        pass
    if temp[5]>=5:
        bits[30]=1;bits[37]=1;bits[44]=1;bits[51]=1;bits[58]=1
    elif temp[5]==4:
        bits[30]=1;bits[37]=1;bits[44]=1;bits[51]=1
    elif temp[5]==3:
        bits[30]=1;bits[37]=1;bits[44]=1
    elif temp[5]==2:
        bits[30]=1;bits[37]=1
    elif temp[5]==1:
        bits[30]=1
    else:
        pass
    if temp[6]>=5:
        bits[65]=1;bits[72]=1;bits[79]=1;bits[86]=1;bits[93]=1
    elif temp[6]==4:
        bits[65]=1;bits[72]=1;bits[79]=1;bits[86]=1
    elif temp[6]==3:
        bits[65]=1;bits[72]=1;bits[79]=1
    elif temp[6]==2:
        bits[65]=1;bits[72]=1
    elif temp[6]==1:
        bits[65]=1
    else:
        pass
    if temp[7]>=2:
        bits[100]=1;bits[107]=1
    elif temp[7]==1:
        bits[100]=1
    else:
        pass
    if temp[8]>=2:
        bits[114]=1;bits[121]=1
    elif temp[8]==1:
        bits[114]=1
    else:
        pass
    if temp[9]>=1:
        bits[128]=1;
    else:
        pass
    if temp[10]>=1:
        bits[135]=1;
    else:
        pass
    return ringSize, bits


def func_4(mol,bits):
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
        ######## aromatic heteroatom-containing
        aromatic = True
        heteroatom = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [1,6] or EndAtom.GetAtomicNum() not in [1,6]:
                heteroatom = True
                break
        if aromatic == True and heteroatom == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[3]=1;bits[10]=1
    elif temp[3]==1:
        bits[3]=1
    else:
        pass
    if temp[4]>=2:
        bits[17]=1;bits[24]=1
    elif temp[4]==1:
        bits[17]=1
    else:
        pass
    if temp[5]>=5:
        bits[31]=1;bits[38]=1;bits[45]=1;bits[52]=1;bits[59]=1
    elif temp[5]==4:
        bits[31]=1;bits[38]=1;bits[45]=1;bits[52]=1
    elif temp[5]==3:
        bits[31]=1;bits[38]=1;bits[45]=1
    elif temp[5]==2:
        bits[31]=1;bits[38]=1
    elif temp[5]==1:
        bits[31]=1
    else:
        pass
    if temp[6]>=5:
        bits[66]=1;bits[73]=1;bits[80]=1;bits[87]=1;bits[94]=1
    elif temp[6]==4:
        bits[66]=1;bits[73]=1;bits[80]=1;bits[87]=1
    elif temp[6]==3:
        bits[66]=1;bits[73]=1;bits[80]=1
    elif temp[6]==2:
        bits[66]=1;bits[73]=1
    elif temp[6]==1:
        bits[66]=1
    else:
        pass
    if temp[7]>=2:
        bits[101]=1;bits[108]=1
    elif temp[7]==1:
        bits[101]=1
    else:
        pass
    if temp[8]>=2:
        bits[115]=1;bits[122]=1
    elif temp[8]==1:
        bits[115]=1
    else:
        pass
    if temp[9]>=1:
        bits[129]=1;
    else:
        pass
    if temp[10]>=1:
        bits[136]=1;
    else:
        pass
    return ringSize,bits


def func_5(mol,bits):
    ringSize=[]
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        Allcarb = True
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name=='AROMATIC':
                nonaromatic = False
                break
        ######## allcarb
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() != 6 or EndAtom.GetAtomicNum() != 6:
                Allcarb = False
                break
        if unsaturated == True and nonaromatic == True and Allcarb == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[4]=1;bits[11]=1
    elif temp[3]==1:
        bits[4]=1
    else:
        pass
    if temp[4]>=2:
        bits[18]=1;bits[25]=1
    elif temp[4]==1:
        bits[18]=1
    else:
        pass
    if temp[5]>=5:
        bits[32]=1;bits[39]=1;bits[46]=1;bits[53]=1;bits[60]=1
    elif temp[5]==4:
        bits[32]=1;bits[39]=1;bits[46]=1;bits[53]=1
    elif temp[5]==3:
        bits[32]=1;bits[39]=1;bits[46]=1
    elif temp[5]==2:
        bits[32]=1;bits[39]=1
    elif temp[5]==1:
        bits[32]=1
    else:
        pass
    if temp[6]>=5:
        bits[67]=1;bits[74]=1;bits[81]=1;bits[88]=1;bits[95]=1
    elif temp[6]==4:
        bits[67]=1;bits[74]=1;bits[81]=1;bits[88]=1
    elif temp[6]==3:
        bits[67]=1;bits[74]=1;bits[81]=1
    elif temp[6]==2:
        bits[67]=1;bits[74]=1
    elif temp[6]==1:
        bits[67]=1
    else:
        pass
    if temp[7]>=2:
        bits[102]=1;bits[109]=1
    elif temp[7]==1:
        bits[102]=1
    else:
        pass
    if temp[8]>=2:
        bits[116]=1;bits[123]=1
    elif temp[8]==1:
        bits[116]=1
    else:
        pass
    if temp[9]>=1:
        bits[130]=1;
    else:
        pass
    if temp[10]>=1:
        bits[137]=1;
    else:
        pass
    return ringSize,bits


def func_6(mol,bits):
    ringSize=[]
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        ContainNitro = False
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name=='AROMATIC':
                nonaromatic = False
                break
        ######## nitrogen-containing
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() == 7 or EndAtom.GetAtomicNum() == 7:
                ContainNitro = True
                break
        if unsaturated == True and nonaromatic == True and ContainNitro== True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[5]=1;bits[12]=1
    elif temp[3]==1:
        bits[5]=1
    else:
        pass
    if temp[4]>=2:
        bits[19]=1;bits[26]=1
    elif temp[4]==1:
        bits[19]=1
    else:
        pass
    if temp[5]>=5:
        bits[33]=1;bits[40]=1;bits[47]=1;bits[54]=1;bits[61]=1
    elif temp[5]==4:
        bits[33]=1;bits[40]=1;bits[47]=1;bits[54]=1
    elif temp[5]==3:
        bits[33]=1;bits[40]=1;bits[47]=1
    elif temp[5]==2:
        bits[33]=1;bits[40]=1
    elif temp[5]==1:
        bits[33]=1
    else:
        pass
    if temp[6]>=5:
        bits[68]=1;bits[75]=1;bits[82]=1;bits[89]=1;bits[96]=1
    elif temp[6]==4:
        bits[68]=1;bits[75]=1;bits[82]=1;bits[89]=1
    elif temp[6]==3:
        bits[68]=1;bits[75]=1;bits[82]=1
    elif temp[6]==2:
        bits[68]=1;bits[75]=1
    elif temp[6]==1:
        bits[68]=1
    else:
        pass
    if temp[7]>=2:
        bits[103]=1;bits[110]=1
    elif temp[7]==1:
        bits[103]=1
    else:
        pass
    if temp[8]>=2:
        bits[117]=1;bits[124]=1
    elif temp[8]==1:
        bits[117]=1
    else:
        pass
    if temp[9]>=1:
        bits[131]=1;
    else:
        pass
    if temp[10]>=1:
        bits[138]=1;
    else:
        pass
    return ringSize,bits


def func_7(mol,bits):

    ringSize=[]
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        heteroatom = False
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name=='AROMATIC':
                nonaromatic = False
                break
        ######## heteroatom-containing
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [1,6] or EndAtom.GetAtomicNum() not in [1,6]:
                heteroatom = True
                break
        if unsaturated == True and nonaromatic == True and heteroatom == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
    if temp[3]>=2:
        bits[6]=1;bits[13]=1
    elif temp[3]==1:
        bits[6]=1
    else:
        pass
    if temp[4]>=2:
        bits[20]=1;bits[27]=1
    elif temp[4]==1:
        bits[20]=1
    else:
        pass
    if temp[5]>=5:
        bits[34]=1;bits[41]=1;bits[48]=1;bits[55]=1;bits[62]=1
    elif temp[5]==4:
        bits[34]=1;bits[41]=1;bits[48]=1;bits[55]=1
    elif temp[5]==3:
        bits[34]=1;bits[41]=1;bits[48]=1
    elif temp[5]==2:
        bits[34]=1;bits[41]=1
    elif temp[5]==1:
        bits[34]=1
    else:
        pass
    if temp[6]>=5:
        bits[69]=1;bits[76]=1;bits[83]=1;bits[90]=1;bits[97]=1
    elif temp[6]==4:
        bits[69]=1;bits[76]=1;bits[83]=1;bits[90]=1
    elif temp[6]==3:
        bits[69]=1;bits[76]=1;bits[83]=1
    elif temp[6]==2:
        bits[69]=1;bits[76]=1
    elif temp[6]==1:
        bits[69]=1
    else:
        pass
    if temp[7]>=2:
        bits[104]=1;bits[111]=1
    elif temp[7]==1:
        bits[104]=1
    else:
        pass
    if temp[8]>=2:
        bits[118]=1;bits[125]=1
    elif temp[8]==1:
        bits[118]=1
    else:
        pass
    if temp[9]>=1:
        bits[132]=1;
    else:
        pass
    if temp[10]>=1:
        bits[139]=1;
    else:
        pass
    return ringSize,bits


def func_8(mol, bits):

    AllRingsBond = mol.GetRingInfo().BondRings()
    temp={'aromatic':0,'heteroatom':0}
    for ring in AllRingsBond:
        aromatic = True
        heteroatom = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        if aromatic==True:
            temp['aromatic']+=1
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [1,6] or EndAtom.GetAtomicNum() not in [1,6]:
                heteroatom = True
                break
        if heteroatom==True:
            temp['heteroatom']+=1
    if temp['aromatic']>=4:
        bits[140]=1;bits[142]=1;bits[144]=1;bits[146]=1
    elif temp['aromatic']==3:
        bits[140]=1;bits[142]=1;bits[144]=1
    elif temp['aromatic']==2:
        bits[140]=1;bits[142]=1
    elif temp['aromatic']==1:
        bits[140]=1
    else:
        pass
    if temp['aromatic']>=4 and temp['heteroatom']>=4:
        bits[141]=1;bits[143]=1;bits[145]=1;bits[147]=1
    elif temp['aromatic']==3 and temp['heteroatom']==3:
        bits[141]=1;bits[143]=1;bits[145]=1
    elif temp['aromatic']==2 and temp['heteroatom']==2:
        bits[141]=1;bits[143]=1
    elif temp['aromatic']==1 and temp['heteroatom']==1:
        bits[141]=1
    else:
        pass
    return bits


def calcPubChemFingerPart2(mol):# 116-263

    bits=[0]*148
    bits=func_1(mol,bits)[1]
    bits=func_2(mol,bits)[1]
    bits=func_3(mol,bits)[1]
    bits=func_4(mol,bits)[1]
    bits=func_5(mol,bits)[1]
    bits=func_6(mol,bits)[1]
    bits=func_7(mol,bits)[1]
    bits=func_8(mol,bits)

    return bits


def calcPubChemFingerAll(s):
    mol = Chem.MolFromSmiles(s)
    AllBits=[0]*881
    res1=list(calcPubChemFingerPart1(mol).ToBitString())
    for index, item in enumerate(res1[1:116]):
        if item == '1':
            AllBits[index] = 1
    for index2, item2 in enumerate(res1[116:734]):
        if item2 == '1':
            AllBits[index2+115+148] = 1
    res2=calcPubChemFingerPart2(mol)
    for index3, item3 in enumerate(res2):
        if item3==1:
            AllBits[index3+115]=1
    return np.array(AllBits)

def smiles2pubchem(s):
  s = canonicalize(s)
  # try:
  features = calcPubChemFingerAll(s)
  # except:
  #   print('pubchem fingerprint not working for smiles: ' + s + ' convert to 0 vectors')
  #   features = np.zeros((881, ))
  return np.array(features)


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

def selfies2ECFP2(selfies):
  smiles = selfies2smiles(selfies)
  smiles = canonicalize(smiles)
  return smiles2ECFP2(smiles)

def selfies2ECFP4(selfies):
  smiles = selfies2smiles(selfies)
  smiles = canonicalize(smiles)
  return smiles2ECFP4(smiles)

def selfies2ECFP6(selfies):
  smiles = selfies2smiles(selfies)
  smiles = canonicalize(smiles)
  return smiles2ECFP6(smiles)

def selfies2MACCS(selfies):
  smiles = selfies2smiles(selfies)
  smiles = canonicalize(smiles)
  return smiles2maccs(smiles)

def selfies2Daylight(selfies):
  smiles = selfies2smiles(selfies)
  smiles = canonicalize(smiles)
  return smiles2daylight(smiles)


def selfies2RDKit2D(selfies):
  smiles = selfies2smiles(selfies)
  smiles = canonicalize(smiles)
  return smiles2rdkit2d(smiles)


def selfies2Morgan(selfies):
  smiles = selfies2smiles(selfies)
  smiles = canonicalize(smiles)
  return smiles2morgan(smiles)

def selfies2PubChem(selfies):
  smiles = selfies2smiles(selfies)
  smiles = canonicalize(smiles)
  return smiles2pubchem(smiles)



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

def selfies2graph2D(selfies):
  smiles = selfies2smiles(selfies)
  smiles = canonicalize(smiles)
  return smiles2graph2D(smiles)


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

def selfies2PyG(selfies):
  smiles = selfies2smiles(selfies)
  smiles = canonicalize(smiles)
  return smiles2PyG(smiles)

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

def selfies2DGL(selfies):
  smiles = selfies2smiles(selfies)
  smiles = canonicalize(smiles)
  return smiles2DGL(smiles)

def molfile2DGL(molfile):
  smiles = molfile2smiles(molfile)
  smiles = canonicalize(smiles)
  return smiles2DGL(smiles)


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
 

def sdffile2graph3d_lst(sdffile):
  with open(sdffile, 'r') as fin:
    lines = fin.readlines() 
  texts = '\t\t\t'.join(lines)
  graph3d_lst = []
  texts = texts.split('$$$$')
  for single_block in texts:
    if single_block.strip()=='':
      break
    try:
      lines = single_block.split('\t\t\t')
      lines = list(filter(lambda x:len(x.strip())>0, lines))
      # for line in lines:
      #   print(line.strip())
      atoms_feature = []
      bonds_feature = []
      line = lines[1]
      atom_num = int(line.strip().split()[0])
      bond_num = int(line.strip().split()[1]) 
      # if atom_num > 1000:
      #   length = int(len(line.strip().split()[0])/2)
      #   atom_num = int(line.strip().split()[0][:length])
      #   bond_num = int(line.strip().split()[0][length:])
      #   print(atom_num, bond_num)

      atom_lines = lines[2:2+atom_num]
      bond_lines = lines[2+atom_num:2+atom_num+bond_num]
      distance_adj_matrix = np.zeros((atom_num, atom_num))
      bondtype_adj_matrix = np.zeros((atom_num, atom_num), dtype = int)
      idx2atom = {idx:line.strip().split()[3] for idx,line in enumerate(atom_lines)}
      coordinates = [[float(line.strip().split()[0]), float(line.strip().split()[1]), float(line.strip().split()[2])] for line in atom_lines]
      for i in range(atom_num):
        for j in range(i+1, atom_num):
          distance_adj_matrix[i,j] = distance_adj_matrix[j,i] = distance3d(coordinates[i], coordinates[j])
      for line in bond_lines:
        a1 = int(line.strip().split()[0])-1
        a2 = int(line.strip().split()[1])-1
        bondtype = int(line.strip().split()[2])
        bondtype_adj_matrix[a1,a2]=bondtype_adj_matrix[a2,a1] = bondtype
      graph3d_lst.append((idx2atom, distance_adj_matrix, bondtype_adj_matrix))
    except:
      pass 
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


##### mol file
def molfile2smiles(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles 

def molfile2selfies(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2selfies(smiles)

def molfile2graph2d(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2graph2D(smiles)

def molfile2pyg(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2PyG(smiles)

def molfile2dgl(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2DGL(smiles)

def molfile2ecfp2(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2ECFP2(smiles)

def molfile2ecfp4(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2ECFP4(smiles)

def molfile2ecfp6(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2ECFP6(smiles)

def molfile2maccs(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2maccs(smiles)

def molfile2daylight(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2daylight(smiles)

def molfile2rdkit(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2rdkit2d(smiles)

def molfile2morgan(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2morgan(smiles)

def molfile2pubchem(molfile):
  mol = Chem.MolFromMolFile(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2pubchem(smiles)


##### mol2 file 
def mol2file2smiles(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles 

def mol2file2selfies(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  return smiles2selfies(smiles)

def mol2file2graph2d(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2graph2D(smiles)

def mol2file2pyg(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2PyG(smiles)

def mol2file2dgl(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2DGL(smiles)

def mol2file2ecfp2(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2ECFP2(smiles)

def mol2file2ecfp4(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2ECFP4(smiles)

def mol2file2ecfp6(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2ECFP6(smiles)

def mol2file2maccs(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2maccs(smiles)

def mol2file2daylight(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2daylight(smiles)

def mol2file2rdkit(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2rdkit2d(smiles)

def mol2file2morgan(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2morgan(smiles)

def mol2file2pubchem(molfile):
  mol = Chem.MolFromMol2File(molfile)
  smiles = Chem.MolToSmiles(mol)
  smiles = canonicalize(smiles)
  return smiles2pubchem(smiles)

#2D_format = ['SMILES', 'SELFIES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem']
#3D_format = ['Graph3D', 'Coulumb']

convert_dict = {
          'SMILES': ['SELFIES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'],
          'SELFIES': ['SMILES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'], 
          'mol': ['SMILES', 'SELFIES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'],
          'mol2': ['SMILES', 'SELFIES', 'Graph2D', 'PyG', 'DGL', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'], 
          'SDF': ['SMILES', 'SELFIES', 'Graph3D', 'Coulumb'],
          'XYZ': ['SMILES', 'SELFIES', 'Graph3D', 'Coulumb'],  
        }

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

    def __init__(self, src = 'SMILES', dst = 'Graph2D'):
        self._src = src
        self._dst = dst

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

        if src == 'SMILES' and dst == 'SMILES':
          self.func = canonicalize
        elif src == 'SMILES' and dst == 'SELFIES':
          self.func = smiles2selfies
        elif src == 'SMILES' and dst == 'Graph2D':
          self.func = smiles2graph2D
        elif src == 'SMILES' and dst == 'PyG':
          self.func = smiles2PyG 
        elif src == 'SMILES' and dst == 'DGL':
          self.func = smiles2DGL
        elif src == 'SMILES' and dst == 'ECFP2':
          self.func = smiles2ECFP2
        elif src == 'SMILES' and dst == 'ECFP4':
          self.func = smiles2ECFP4 
        elif src == 'SMILES' and dst == 'ECFP6':
          self.func = smiles2ECFP6 
        elif src == 'SMILES' and dst == 'MACCS':
          self.func = smiles2maccs 
        elif src == 'SMILES' and dst == 'Daylight':
          self.func = smiles2daylight 
        elif src == 'SMILES' and dst == 'RDKit2D':
          self.func = smiles2rdkit2d 
        elif src == 'SMILES' and dst == 'Morgan':
          self.func = smiles2morgan 
        elif src == 'SMILES' and dst == 'PubChem':
          self.func = smiles2pubchem 

        #### SELFIES 
        elif src == 'SELFIES' and dst == 'SMILES':
          self.func = selfies2smiles
        elif src == 'SELFIES' and dst == 'Graph2D':
          self.func = selfies2graph2D
        elif src == 'SELFIES' and dst == 'PyG':
          self.func = selfies2PyG 
        elif src == 'SELFIES' and dsst == 'DGL':
          self.func = selfies2DGL
        elif src == 'SELFIES' and dst == 'ECFP2':
          self.func = selfies2ECFP2
        elif src == 'SELFIES' and dst == 'ECFP4':
          self.func = selfies2ECFP4
        elif src == 'SELFIES' and dst == 'ECFP6':
          self.func = selfies2ECFP6
        elif src == 'SELFIES' and dst == 'MACCS':
          self.func = selfies2MACCS
        elif src == 'SELFIES' and dst == 'Daylight':
          self.func = selfies2Daylight
        elif src == 'SELFIES' and dst == 'RDKit2D':
          self.func = selfies2RDKit2D
        elif src == 'SELFIES' and dst == 'Morgan':
          self.func = selfies2Morgan
        elif src == 'SELFIES' and dst == 'PubChem':
          self.func = selfies2PubChem

        ### load from xyz file, input is a filename (str), only contain one smiles 
        elif src == 'XYZ' and dst == 'SMILES':
          self.func = xyzfile2smiles
        elif src == 'XYZ' and dst == 'SELFIES':
          self.func = xyzfile2selfies 
        elif src == 'XYZ' and dst == 'Graph3D':
          self.func = xyzfile2graph3d 
        elif src == 'XYZ' and dst == 'Coulomb':
          self.func = xyzfile2coulomb 

        ### SDF file 
        elif src == 'SDF' and dst == 'Graph3D':
          self.func = sdffile2graph3d_lst 
        elif src == 'SDF' and dst == 'SMILES':
          self.func = sdffile2smiles_lst  
        elif src == 'SDF' and dst == 'SELFIES':
          self.func = sdffile2selfies_lst 
        elif src == 'SDF' and dst == 'Coulomb':
          self.func = sdffile2coulomb

        ### mol file
        elif src == 'mol' and dst == 'SMILES':
          self.func = molfile2smiles 
        elif src == 'mol' and dst == 'SELFIES':
          self.func = molfile2selfies 
        elif src == 'mol' and dst == 'Graph2D':
          self.func = molfile2graph2d 
        elif src == 'mol' and dst == 'ECFP2':
          self.func = molfile2ecfp2 
        elif src == 'mol' and dst == 'ECFP4':
          self.func = molfile2ecfp4 
        elif src == 'mol' and dst == 'ECFP6':
          self.func = molfile2ecfp6 
        elif src == 'mol' and dst == 'MACCS':
          self.func = molfile2maccs 
        elif src == 'mol' and dst == 'Daylight':
          self.func = molfile2daylight
        elif src == 'mol' and dst == 'RDKit2D':
          self.func = molfile2rdkit
        elif src == 'mol' and dst == 'Morgan':
          self.func = molfile2morgan 
        elif src == 'mol' and dst == 'PubChem':
          self.func = molfile2pubchem  
        # todo 'mol': ['SMILES', 'SELFIES', 'Graph2D', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'],


        ### mol2 file
        elif src == 'mol2' and dst == 'SMILES':
          self.func = mol2file2smiles 
        elif src == 'mol2' and dst == 'SELFIES':
          self.func = mol2file2selfies 
        elif src == 'mol2' and dst == 'Graph2D':
          self.func = mol2file2graph2d 
        elif src == 'mol2' and dst == 'ECFP2':
          self.func = mol2file2ecfp2 
        elif src == 'mol2' and dst == 'ECFP4':
          self.func = mol2file2ecfp4 
        elif src == 'mol2' and dst == 'ECFP6':
          self.func = mol2file2ecfp6 
        elif src == 'mol2' and dst == 'MACCS':
          self.func = mol2file2maccs 
        elif src == 'mol2' and dst == 'Daylight':
          self.func = mol2file2daylight
        elif src == 'mol2' and dst == 'RDKit2D':
          self.func = mol2file2rdkit
        elif src == 'mol2' and dst == 'Morgan':
          self.func = mol2file2morgan 
        elif src == 'mol2' and dst == 'PubChem':
          self.func = mol2file2pubchem     
        # todo 'mol2': ['SMILES', 'SELFIES', 'Graph2D', 'ECFP2', 'ECFP4', 'ECFP6', 'MACCS', 'Daylight', 'RDKit2D', 'Morgan', 'PubChem'],




    def __call__(self, x):
      if type(x) == str:
        return self.func(x)
      elif type(x) == list:
        return list(map(self.func, x))


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


