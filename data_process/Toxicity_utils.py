import pandas as pd
import numpy as np
import os, sys, json, wget, subprocess
import warnings
warnings.filterwarnings("ignore")

from .. import utils

def Tox21_process(name, path, target = None):

	utils.download_unzip(name, path, 'tox21.csv')

	df = pd.read_csv(os.path.join(path,'tox21.csv'))
	
	df = df[df[target].notnull()].reset_index(drop = True)
	df = df.iloc[df['smiles'].drop_duplicates(keep = False).index.values]

	y = df[target].values
	drugs = df.smiles.values
	drugs_idx = df["mol_id"].values

	return drugs, y, drugs_idx

def ToxCast_process(name, path, target = None):

	utils.download_unzip(name, path, 'toxcast_data.csv')

	df = pd.read_csv(os.path.join(path,'toxcast_data.csv'))
	
	df = df[df[target].notnull()].reset_index(drop = True)
	df = df.iloc[df['smiles'].drop_duplicates(keep = False).index.values]

	y = df[target].values
	drugs = df.smiles.values
	drugs_idx = np.array(['Drug ' + str(i) for i in list(range(len(drugs)))])

	return drugs, y, drugs_idx

def ClinTox_process(name, path, target = None):

	utils.download_unzip(name, path, 'clintox.csv')
	df = pd.read_csv(os.path.join(path,'clintox.csv'))
	df = df.iloc[df['smiles'].drop_duplicates(keep = False).index.values]
	
	y = df["CT_TOX"].values
	drugs = df.smiles.values
	drugs_idx = np.array(['Drug ' + str(i) for i in list(range(len(drugs)))])

	return drugs, y, drugs_idx