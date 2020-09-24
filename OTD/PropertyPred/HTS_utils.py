import pandas as pd
import numpy as np
import os, sys, json, wget
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from .. import utils


def cid2smiles(cid, cid_dict):
	cid = int(cid)
	if cid in cid_dict:
		return cid_dict[cid]
	else:
		return utils.cid2smiles(cid)

def PubChemAID_process(name, path, target = None):
	if target is None:
		raise AttributeError('Please specify an AID BioAssay Index')
	else:
		if not os.path.exists(os.path.join(path,'AID_' + str(target) + '_datatable_all.csv')):
			utils.download('https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&actvty=all&response_type=save&aid='+str(target), path)

	df = pd.read_csv(os.path.join(path,'AID_' + str(target) + '_datatable_all.csv'))[['PUBCHEM_CID', 'PUBCHEM_ACTIVITY_OUTCOME']].drop_duplicates(keep = False)
	df = df[df.PUBCHEM_CID.notnull()]
	utils.print_sys("Retrieving SMILES strings for the drugs...")

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/cid2smiles.pkl', path)
	cid_dict = utils.load_dict(os.path.join(path, 'cid2smiles.pkl'))
	smiles = [cid2smiles(i, cid_dict) for i in tqdm(df.PUBCHEM_CID.values)]

	# label mapping
	label_set = np.unique(df.PUBCHEM_ACTIVITY_OUTCOME.values)
	label_index = list(range(len(label_set)))
	label_dict = dict(zip(label_set, label_index))
	df['Label'] = df.PUBCHEM_ACTIVITY_OUTCOME.apply(lambda x: label_dict[x])

	df = df.reset_index(drop = True)
	df.to_csv(os.path.join(path,'AID_' + str(target) + '_datatable_all.csv'))
	y = df.Label.values
	drugs = np.array(smiles)
	drugs_idx = df.PUBCHEM_CID.values

	return drugs, y, drugs_idx

def PCBA_process(name, path, target = None):
	if target is None:
		raise AttributeError('Please specify a PCBA target Index from target_list.py')

	utils.download_unzip(name, path, 'pcba.csv')

	df = pd.read_csv(os.path.join(path,'pcba.csv'))
	df = df.iloc[df['smiles'].drop_duplicates(keep = False).index.values]

	df = df[df[str(target)].notnull()].reset_index(drop = True)
	y = df[str(target)].values
	drugs = df.smiles.values
	drugs_idx = df.mol_id.values

	return drugs, y, drugs_idx

def MUV_process(name, path, target = None):
	if target is None:
		raise AttributeError('Please specify a MUV target Index from target_list.py')

	utils.download_unzip(name, path, 'muv.csv')

	df = pd.read_csv(os.path.join(path,'muv.csv'))
	df = df.iloc[df['smiles'].drop_duplicates(keep = False).index.values]

	df = df[df[str(target)].notnull()].reset_index(drop = True)
	y = df[str(target)].values
	drugs = df.smiles.values
	drugs_idx = df.mol_id.values

	return drugs, y, drugs_idx

def HIV_process(name, path, target = None):
	utils.download_unzip(name, path, 'hiv.csv')

	df = pd.read_csv(os.path.join(path,'hiv.csv'))
	df = df.iloc[df['smiles'].drop_duplicates(keep = False).index.values]

	df = df[df["HIV_active"].notnull()].reset_index(drop = True)
	y = df["HIV_active"].values
	drugs = df.smiles.values
	drugs_idx = np.array(list(range(len(drugs))))

	return drugs, y, drugs_idx	

def BACE_process(name, path, target = None):
	if (target is None) or (target not in ['pIC50', 'class']):
		raise AttributeError('Choose target either pIC50 or class.')

	utils.download_unzip(name, path, 'bace.csv')

	df = pd.read_csv(os.path.join(path,'bace.csv'))
	df = df.iloc[df['mol'].drop_duplicates(keep = False).index.values]

	df = df[df[target].notnull()].reset_index(drop = True)
	y = df[target].values
	drugs = df.mol.values
	drugs_idx = np.array(list(range(len(drugs))))

	return drugs, y, drugs_idx	


def CathepsinS_process(name, path, target = None):
	
	utils.download_unzip(name, path, 'CatS')

	df = pd.read_csv(os.path.join(path, 'CatS/CatS_score_compounds_D3R_GC4_answers.csv'))

	y = df['Affinity'].values
	drugs = df.SMILES.values
	drugs_idx = df.Cmpd_ID.values

	return drugs, y, drugs_idx

def SARS_CoV_3CLPro_process(name, path, target = None):
	url = 'https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&actvty=all&response_type=save&aid=1706'
	saved_path_data = wget.download(url, path)

	url = 'https://drive.google.com/uc?export=download&id=1eipPaFrg-mVULoBhyp2kvEemi2WhDxsM'
	saved_path_conversion = wget.download(url, path)

	df_data = pd.read_csv(saved_path_data)
	df_conversion = pd.read_csv(saved_path_conversion)
	val = df_data.iloc[4:][['PUBCHEM_CID','PUBCHEM_ACTIVITY_SCORE']]

	cid2smiles = dict(zip(df_conversion[['cid','smiles']].values[:, 0], df_conversion[['cid','smiles']].values[:, 1]))
	X_drug = [cid2smiles[i] for i in val.PUBCHEM_CID.values]
	
	print("AID1706 is usually binarized, you can binarize by 'data.binarize(threshold = 15, order = 'ascending')', and threshold is recommended to be 15", flush = True, file = sys.stderr)
	y = val.PUBCHEM_ACTIVITY_SCORE.values

	y = np.array(y)
	drugs = np.array(X_drug)
	drugs_idx = val.PUBCHEM_CID.values

	return drugs, y, drugs_idx