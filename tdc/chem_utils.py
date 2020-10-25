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
import networkx as nx 
from .drd2_scorer import get_score as drd2 

## from https://github.com/wengong-jin/iclr19-graph2graph/blob/master/props/properties.py 

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









if __name__ == "__main__":
	smiles = '[H][C@@]12C[C@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'
	print(similarity(smiles, smiles))
	print(qed(smiles))
	print(penalized_logp(smiles))
	print(drd2(smiles))

#  conda install -c rdkit rdkit






