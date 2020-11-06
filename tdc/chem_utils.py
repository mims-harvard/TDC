try: 
	import rdkit
	from rdkit import Chem, DataStructs
	from rdkit.Chem import AllChem
	from rdkit.Chem import Descriptors
	import rdkit.Chem.QED as QED
except:
	raise ImportError("Please install rdkit by 'conda install -c conda-forge rdkit'! ")	

# import sascorer
from .sascorer import * 
try:
	import networkx as nx 
except:
	raise ImportError("Please install networkx by 'pip install networkx'! ")	
from .drd2_scorer import get_score as drd2 
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
	return novel_ratio

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

if __name__ == "__main__":
	smiles = '[H][C@@]12C[C@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'
	print(similarity(smiles, smiles))
	print(qed(smiles))
	print(penalized_logp(smiles))
	print(drd2(smiles))
	print(SA(smiles))
	list_of_smiles = ['CCC', 'fewjio', smiles, smiles]
	print(validity_ratio(list_of_smiles))
	print(unique_rate(list_of_smiles))
	#  conda install -c rdkit rdkit






