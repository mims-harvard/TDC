import pandas as pd
import numpy as np
from tqdm import tqdm
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .. import utils

def DAVIS_process(name, path, target = None):
	
	utils.download_unzip(name, path, 'DAVIS')
	affinity = pd.read_csv(path + '/DAVIS/affinity.txt', header=None, sep = ' ')

	with open(path + '/DAVIS/target_seq.txt') as f:
		target = list(json.load(f).values())

	with open(path + '/DAVIS/SMILES.txt') as f:
		drug = list(json.load(f).values())

	SMILES, Target_seq, y = [], [], []

	for i in tqdm(range(len(drug))):
		for j in range(len(target)):
			SMILES.append(drug[i])
			Target_seq.append(target[j])
			y.append(affinity.values[i, j])

	y = np.array(y)
	targets = np.array(Target_seq)
	drugs = np.array(SMILES)

	drugs_idx_map = dict(zip(np.unique(drugs), list(range(len(np.unique(drugs))))))
	targets_idx_map = dict(zip(np.unique(targets), list(range(len(np.unique(targets))))))

	drugs_idx = np.array(['Drug ' + str(drugs_idx_map[i]) for i in drugs])
	targets_idx = np.array(['Target ' + str(targets_idx_map[i]) for i in targets])

	return drugs, targets, y, drugs_idx, targets_idx

def KIBA_process(name, path, target = None):

	utils.download_unzip(name, path, 'KIBA')

	affinity = pd.read_csv(path + '/KIBA/affinity.txt', header=None, sep = '\t')
	affinity = affinity.fillna(-1)

	with open(path + '/KIBA/target_seq.txt') as f:
		target = json.load(f)

	with open(path + '/KIBA/SMILES.txt') as f:
		drug = json.load(f)

	target = list(target.values())
	drug = list(drug.values())

	SMILES = []
	Target_seq = []
	y = []

	for i in tqdm(range(len(drug))):
		for j in range(len(target)):
			if affinity.values[i, j] != -1:
				SMILES.append(drug[i])
				Target_seq.append(target[j])
				y.append(affinity.values[i, j])

	y = np.array(y)
	targets = np.array(Target_seq)
	drugs = np.array(SMILES)

	drugs_idx_map = dict(zip(np.unique(drugs), list(range(len(np.unique(drugs))))))
	targets_idx_map = dict(zip(np.unique(targets), list(range(len(np.unique(targets))))))

	drugs_idx = np.array(['Drug ' + str(drugs_idx_map[i]) for i in drugs])
	targets_idx = np.array(['Target ' + str(targets_idx_map[i]) for i in targets])

	return drugs, targets, y, drugs_idx, targets_idx

def KIBA_process(name, path, target = None):

	utils.download_unzip(name, path, 'KIBA')

	affinity = pd.read_csv(path + '/KIBA/affinity.txt', header=None, sep = '\t')
	affinity = affinity.fillna(-1)

	with open(path + '/KIBA/target_seq.txt') as f:
		target = json.load(f)

	with open(path + '/KIBA/SMILES.txt') as f:
		drug = json.load(f)

	target = list(target.values())
	drug = list(drug.values())

	SMILES = []
	Target_seq = []
	y = []

	for i in tqdm(range(len(drug))):
		for j in range(len(target)):
			if affinity.values[i, j] != -1:
				SMILES.append(drug[i])
				Target_seq.append(target[j])
				y.append(affinity.values[i, j])

	y = np.array(y)
	targets = np.array(Target_seq)
	drugs = np.array(SMILES)

	drugs_idx_map = dict(zip(np.unique(drugs), list(range(len(np.unique(drugs))))))
	targets_idx_map = dict(zip(np.unique(targets), list(range(len(np.unique(targets))))))

	drugs_idx = np.array(['Drug ' + str(drugs_idx_map[i]) for i in drugs])
	targets_idx = np.array(['Target ' + str(targets_idx_map[i]) for i in targets])

	return drugs, targets, y, drugs_idx, targets_idx


def BindingDB_process(name, path, target = None):
	if target is None:
		utils.print_sys("Default is Kd, you can select from Kd, IC50, EC50, Ki.")
		target = 'Kd'

	utils.download_unzip('BindingDB_' + target, path, 'BindingDB_' + target + '.csv')

	df = pd.read_csv(os.path.join(path,'BindingDB_' + target + '.csv'))

	return df.SMILES.values, df['Target Sequence'].values, df.Label.values, df.PubChem_ID.values, df.UniProt_ID.values, 
