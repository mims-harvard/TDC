import pandas as pd
import numpy as np
import os, sys, json, wget, subprocess
import warnings
warnings.filterwarnings("ignore")

from .. import utils

def Lipo_AZ_process(name, path, target = None):
	utils.download_unzip(name, path, 'Lipophilicity.csv')

	df = pd.read_csv(os.path.join(path,'Lipophilicity.csv'))
	df = df.iloc[df['smiles'].drop_duplicates(keep = False).index.values]
	y = df.exp.values
	drugs = df.smiles.values
	drugs_idx = df.CMPD_CHEMBLID.values

	return drugs, y, drugs_idx


def ESOL_process(name, path, target = None):

	utils.download_unzip(name, path, 'delaney-processed.csv')

	df = pd.read_csv(os.path.join(path,'delaney-processed.csv'))
	df = df.iloc[df['smiles'].drop_duplicates(keep = False).index.values]
	
	y = df["measured log solubility in mols per litre"].values
	drugs = df.smiles.values
	drugs_idx = df["Compound ID"].values

	return drugs, y, drugs_idx

def FreeSolv_process(name, path, target = None):

	utils.download_unzip(name, path, 'SAMPL.csv')

	df = pd.read_csv(os.path.join(path,'SAMPL.csv'))
	df = df.iloc[df['smiles'].drop_duplicates(keep = False).index.values]
	y = df["expt"].values
	drugs = df.smiles.values
	drugs_idx = df["iupac"].values

	return drugs, y, drugs_idx

def BBB_MolNet_process(name, path, target = None):

	utils.download_unzip(name, path, 'bbbp.csv')

	df = pd.read_csv(os.path.join(path,'bbbp.csv'))
	df = df.iloc[df['smiles'].drop_duplicates(keep = False).index.values]
	
	y = df["p_np"].values
	drugs = df.smiles.values
	drugs_idx = df["name"].values

	return drugs, y, drugs_idx

def AqSolDB_process(name, path, target = None):

	if os.path.exists(os.path.join(path,'curated-solubility-dataset.csv')):
		print('Dataset already downloaded in the local system...', flush = True, file = sys.stderr)
	else:
		wget.download('https://dataverse.harvard.edu/api/access/datafile/3407241?format=original&gbrecs=true', path)

	df = pd.read_csv(os.path.join(path,'curated-solubility-dataset.csv'))
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]
	
	y = df["Solubility"].values
	drugs = df.SMILES.values
	drugs_idx = df.Name.values

	return drugs, y, drugs_idx

def LogS_process(name, path, target = None):

	utils.install('xlrd')
	utils.S3_download('https://drugdataloader.s3.amazonaws.com/table11.xls', path)

	df = pd.read_excel(os.path.join(path,'table11.xls'), skiprows=2)
	df = df[df.Name.notnull()]
	df = df.iloc[df['Smiles'].drop_duplicates(keep = False).index.values]
	
	y = df["Expt."].values
	drugs = df.Smiles.values
	drugs_idx = df.Name.values

	return drugs, y, drugs_idx

def LogD74_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/logd74.tsv', path)
	df = pd.read_csv(os.path.join(path,'logd74.tsv'), sep='\t')
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]
	y = df["logD7.4"].values
	drugs = df.SMILES.values
	drugs_idx = df.ID.values

	return drugs, y, drugs_idx

def Caco2_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/caco-2.csv', path)

	df = pd.read_csv(os.path.join(path,'caco-2.csv'))
	df = df.iloc[df['smi'].drop_duplicates(keep = False).index.values]
	
	y = df["logPapp"].values
	drugs = df.smi.values
	drugs_idx = df.name.values

	return drugs, y, drugs_idx

def HIA_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/HIA.csv', path)
	#print("Threshold of 30 is recommended for binarization.", flush = True, file = sys.stderr)

	df = pd.read_csv(os.path.join(path,'HIA.csv'))
	df = df.iloc[df['Structure'].drop_duplicates(keep = False).index.values]
	
	y = df["HIA (%)"].values
	y = utils.binarize(y, 30, 'ascending')
	drugs = df.Structure.values
	drugs_idx = df.Molecule_name.values

	return drugs, y, drugs_idx

def BBB_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/BBB.csv', path)

	df = pd.read_csv(os.path.join(path,'BBB.csv'))
	df = df.iloc[df['Structure'].drop_duplicates(keep = False).index.values]
	
	y = df["Class"].values
	drugs = df.Structure.values
	drugs_idx = df.Name.values

	return drugs, y, drugs_idx

def Pgp_inhibitor_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/Pgp_inhibitor.csv', path)

	df = pd.read_csv(os.path.join(path,'Pgp_inhibitor.csv'))
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]
	
	y = df["Activity"].values
	drugs = df.SMILES.values
	drugs_idx = df.Name.values

	return drugs, y, drugs_idx

def PPBR_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/PPBR.csv', path)

	df = pd.read_csv(os.path.join(path,'PPBR.csv'))
	df = df[df.SMILES.apply(lambda x: isinstance(x, str))]	
	df = df.reset_index(drop = True)

	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]
	y = df["Class"].values
	y = np.array([1 if i == 1 else 0 for i in y])
	drugs = df.SMILES.values
	drugs_idx = df["Drug Name"].values

	return drugs, y, drugs_idx

def Bioavailability_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/bioavailability.csv', path)

	df = pd.read_csv(os.path.join(path,'bioavailability.csv'))
	df = df[df.SMILES.apply(lambda x: isinstance(x, str))]
	df = df.reset_index(drop = True)
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]

	y = df["Class"].values
	y = np.array([1 if i == 1 else 0 for i in y])
	drugs = df.SMILES.values
	drugs_idx = df["Drug Name"].values

	return drugs, y, drugs_idx

def Bioavailability_F30_eDrug3D_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/bioavailability_eDrug3D.csv', path)

	df = pd.read_csv(os.path.join(path,'bioavailability_eDrug3D.csv'))
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]

	y = df[" F(percentage) "].values
	drugs = df.SMILES.values
	drugs_idx = df[" Name "].values

	y_ = []
	drugs_ = []
	drugs_idx_ = []

	for i, yi in enumerate(y):
		try:
			j = float(yi)
			y_.append(j)
			drugs_idx_.append(drugs_idx[i])
			drugs_.append(drugs[i])
		except:
			continue
	y = utils.binarize(np.array(y_), 30, 'ascending')
	return np.array(drugs_), y, np.array(drugs_idx_)

def Bioavailability_F20_eDrug3D_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/bioavailability_eDrug3D.csv', path)

	df = pd.read_csv(os.path.join(path,'bioavailability_eDrug3D.csv'))
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]

	y = df[" F(percentage) "].values
	drugs = df.SMILES.values
	drugs_idx = df[" Name "].values

	y_ = []
	drugs_ = []
	drugs_idx_ = []

	for i, yi in enumerate(y):
		try:
			j = float(yi)
			y_.append(j)
			drugs_idx_.append(drugs_idx[i])
			drugs_.append(drugs[i])
		except:
			continue
	y = utils.binarize(np.array(y_), 20, 'ascending')
	return np.array(drugs_), y, np.array(drugs_idx_)

def Clearance_eDrug3D_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/Clearance_eDrug3D.csv', path)

	df = pd.read_csv(os.path.join(path,'Clearance_eDrug3D.csv'))
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]

	y = df[" Cl(liter/hour) "].values
	drugs = df.SMILES.values
	drugs_idx = df[" Name "].values

	y_ = []
	drugs_ = []
	drugs_idx_ = []

	for i, yi in enumerate(y):
		try:
			j = float(yi)
			y_.append(j)
			drugs_idx_.append(drugs_idx[i])
			drugs_.append(drugs[i])
		except:
			continue

	return np.array(drugs_), np.array(y_), np.array(drugs_idx_)

def Half_life_eDrug3D_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/half_life_eDrug3D.csv', path)

	df = pd.read_csv(os.path.join(path,'half_life_eDrug3D.csv'))
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]

	y = df[" t1/2(hour) "].values
	drugs = df.SMILES.values
	drugs_idx = df[" Name "].values

	y_ = []
	drugs_ = []
	drugs_idx_ = []

	for i, yi in enumerate(y):
		try:
			j = float(yi)
			y_.append(j)
			drugs_idx_.append(drugs_idx[i])
			drugs_.append(drugs[i])
		except:
			continue

	return np.array(drugs_), np.array(y_), np.array(drugs_idx_)

def VD_eDrug3D_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/VD_eDrug3D.csv', path)

	df = pd.read_csv(os.path.join(path,'VD_eDrug3D.csv'))
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]

	y = df[" VD(liter) "].values
	drugs = df.SMILES.values
	drugs_idx = df[" Name "].values

	y_ = []
	drugs_ = []
	drugs_idx_ = []

	for i, yi in enumerate(y):
		try:
			j = float(yi)
			y_.append(j)
			drugs_idx_.append(drugs_idx[i])
			drugs_.append(drugs[i])
		except:
			continue

	return np.array(drugs_), np.array(y_), np.array(drugs_idx_)

def PPBR_eDrug3D_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/PPB_eDrug3D.csv', path)

	df = pd.read_csv(os.path.join(path,'PPB_eDrug3D.csv'))
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]

	y = df[" PPB(percentage) "].values
	drugs = df.SMILES.values
	drugs_idx = df[" Name "].values

	y_ = []
	drugs_ = []
	drugs_idx_ = []

	for i, yi in enumerate(y):
		try:
			j = float(yi)
			y_.append(j)
			drugs_idx_.append(drugs_idx[i])
			drugs_.append(drugs[i])
		except:
			continue

	return np.array(drugs_), np.array(y_), np.array(drugs_idx_)

def CYP2C19_process(name, path, target = None):
	df = pd.read_csv('./data/CYP2C19.csv')
	df['p450-cyp2c19-Potency'].fillna(10000, inplace = True)
	df = df.reset_index(drop = True)
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]
	y = df["Label"].values
	drugs = df.SMILES.values
	drugs_idx = df["PUBCHEM_CID"].values

	return drugs, y, drugs_idx

def CYP2D6_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/CYP2D6.csv', path)
	df = pd.read_csv(os.path.join(path,'CYP2D6.csv'))
	df['p450-cyp2d6-Potency'].fillna(10000, inplace = True)
	df = df.reset_index(drop = True)
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]
	y = df["Label"].values
	drugs = df.SMILES.values
	drugs_idx = df["PUBCHEM_CID"].values

	return drugs, y, drugs_idx

def CYP3A4_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/CYP3A4.csv', path)
	df = pd.read_csv(os.path.join(path,'CYP3A4.csv'))
	df['p450-cyp3a4-Potency'].fillna(10000, inplace = True)
	df = df.reset_index(drop = True)
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]
	y = df["Label"].values
	drugs = df.SMILES.values
	drugs_idx = df["PUBCHEM_CID"].values

	return drugs, y, drugs_idx

def CYP1A2_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/CYP1A2.csv', path)
	df = pd.read_csv(os.path.join(path,'CYP1A2.csv'))
	df['p450-cyp1a2-Potency'].fillna(10000, inplace = True)
	df = df.reset_index(drop = True)
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]
	y = df["Label"].values
	drugs = df.SMILES.values
	drugs_idx = df["PUBCHEM_CID"].values

	return drugs, y, drugs_idx

def CYP2C9_process(name, path, target = None):

	utils.S3_download('https://drugdataloader.s3.amazonaws.com/CYP2C9.csv', path)
	df = pd.read_csv(os.path.join(path,'CYP2C9.csv'))
	df['p450-cyp2c9-Potency'].fillna(10000, inplace = True)
	df = df.reset_index(drop = True)
	df = df.iloc[df['SMILES'].drop_duplicates(keep = False).index.values]
	y = df["Label"].values
	drugs = df.SMILES.values
	drugs_idx = df["PUBCHEM_CID"].values

	return drugs, y, drugs_idx