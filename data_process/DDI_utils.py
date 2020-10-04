def TWOSIDES_process():
	import pandas as pd
	df = pd.read_csv('/Users/kexinhuang/Downloads/TWOSIDES.csv')

	import numpy as np
	len(np.unique(df[['# STITCH 1', 'STITCH 2']].values.reshape(-1,)))

	from tdc.utils import cid2smiles
	from tqdm import tqdm
	smiles = []
	for i in tqdm(np.unique(df[['# STITCH 1', 'STITCH 2']].values.reshape(-1,))):
	    try:
	        smiles.append(cid2smiles(int(i.split('CID')[-1])))
	    except:
	        smiles.append('ERROR')
	        print(i)

	id2smiles = dict(zip(np.unique(df[['# STITCH 1', 'STITCH 2']].values.reshape(-1,)), smiles))

	df['X1'] = [id2smiles[i] for i in df['# STITCH 1'].values]
	df['X2'] = [id2smiles[i] for i in df['STITCH 2'].values]

	df = df.rename(columns = {'# STITCH 1': 'X1', 'STITCH 2': 'X2', 'Polypharmacy Side Effect': 'Y'})

	y_unique = np.unique(df.Y.values)
	c2y = dict(zip(y_unique, list(range(len(y_unique)))))
	df['Y'] = [c2y[i] for i in df['Y'].values]
	df = df.rename({'Side Effect Name': 'Map'})

	df.to_csv('./data/data_clean/TWOSIDES.csv', index = False)


def DrugBank_process():
	df_db = pd.read_csv('/Users/kexinhuang/Desktop/Research/temp_repo/deepddi/data/KnownDDI.csv')
	effect = pd.read_csv('/Users/kexinhuang/Desktop/Research/temp_repo/deepddi/data/Interaction_information.csv')
	label2effect = dict(zip(effect['Interaction type'].values, effect['Description'].values))

	df_db['Map'] = [label2effect[i] for i in df_db.Label.values]
	df_db = df_db.rename(columns = {'Drug1': 'ID1', 'Drug2': 'ID2', 'Label': 'Y'})
	db_smiles = pd.read_csv('/Users/kexinhuang/Downloads/structure links 3.csv')
	dbid2smiles = dict(zip(db_smiles['DrugBank ID'].values, db_smiles['SMILES'].values))
	dbid2smiles['DB09323'] = 'O.O.O.O.C(CNCC1=CC=CC=C1)NCC1=CC=CC=C1.[H][C@]12SC(C)(C)[C@@H](N1C(=O)[C@H]2NC(=O)CC1=CC=CC=C1)C(O)=O.[H][C@]12SC(C)(C)[C@@H](N1C(=O)[C@H]2NC(=O)CC1=CC=CC=C1)C(O)=O'
	dbid2smiles['DB13450'] = '[O-]S(=O)(=O)C1=CC=CC=C1.[O-]S(=O)(=O)C1=CC=CC=C1.COC1=CC2=C(C=C1OC)[C@@H](CC1=CC(OC)=C(OC)C=C1)[N@@+](C)(CCC(=O)OCCCCCOC(=O)CC[N@@+]1(C)CCC3=C(C=C(OC)C(OC)=C3)[C@H]1CC1=CC(OC)=C(OC)C=C1)CC2'
	dbid2smiles['DB09396'] = 'O.OS(=O)(=O)C1=CC2=CC=CC=C2C=C1.CCC(=O)O[C@@](CC1=CC=CC=C1)([C@H](C)CN(C)C)C1=CC=CC=C1'
	dbid2smiles['DB09162'] = '[Fe+3].OC(CC([O-])=O)(CC([O-])=O)C([O-])=O'
	dbid2smiles['DB11106'] = 'CC(C)(N)CO.CN1C2=C(NC(Br)=N2)C(=O)N(C)C1=O'
	df_db['X1'] = [dbid2smiles[i] for i in df_db.ID1.values]
	df_db['X2'] = [dbid2smiles[i] for i in df_db.ID2.values]
	df_db.to_csv('./data/data_clean/DrugBank.csv', index = False)