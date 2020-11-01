import pandas as pd

def load_GDSC12():
	# GDSC
	df = pd.read_excel('/Users/kexinhuang/Downloads/drug_response_synergy/mono/GDSC2_fitted_dose_response_25Feb20.xlsx')
	len(df)
	df = df[['COSMIC_ID', 'CELL_LINE_NAME', 'DRUG_ID', 'DRUG_NAME', 'LN_IC50']]
	df_drug = pd.read_csv('/Users/kexinhuang/Downloads/drug_response_synergy/mono/GDSC2_drug.csv')
	df_drug = df_drug[['drug_id', 'pubchem']]
	# that has Pubchem ID
	df_drug = df_drug[df_drug.pubchem != '-']
	df_drug = df_drug[df_drug.pubchem != 'none']
	df_drug = df_drug[df_drug.pubchem != 'several']
	# some drug id associate with multiple pubchem CIDs, just remove them
	df_drug = df_drug[df_drug.pubchem.str.split(',').apply(len) == 1]
	# that has Drug ID
	df = df[df.DRUG_ID.isin(df_drug.drug_id.values)].reset_index(drop = True)
	drugid2pubchemid = dict(df_drug.values)
	# download from https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Pathway_Activity_Scores.html
	df_genome = pd.read_csv('/Users/kexinhuang/Downloads/drug_response_synergy/mono/GDSC_Genomic_RMA_EXP.txt', sep = '\t')
	df_genome.columns = [i[5:] if i[:3] == 'DAT' else i for i in df_genome.columns.values]
	genome_seq = df_genome.iloc[:, 2:].values.T
	cosmic2seq = dict(zip(list(df_genome.columns[2:]), genome_seq))
	df.loc[:, 'COSMIC_ID'] = [str(i) for i in df.COSMIC_ID.values]
	df = df[df.COSMIC_ID.isin(list(df_genome.columns[2:]))]
	df['Pubchem'] = [drugid2pubchemid[i] for i in df['DRUG_ID']]

	# use this to https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi
	for i in list(drugid2pubchemid.values()):
	    print(i)
	df_smiles = pd.read_csv('/Users/kexinhuang/Downloads/drug_response_synergy/mono/GSDC2_SMILES.txt', sep = '\t', header = None)
	df_smiles = df_smiles[~df_smiles[1].isnull()]
	df_smiles[0] = [str(i) for i in df_smiles[0].values]

	# that has SMILES
	df = df[df.Pubchem.isin(df_smiles[0].values)]
	df['SMILES'] = [dict(df_smiles.values)[i] for i in df.Pubchem]

	df['Expression'] = [cosmic2seq[i] for i in df.COSMIC_ID]

	df = df[['DRUG_NAME', 'CELL_LINE_NAME', 'SMILES', 'Expression', 'LN_IC50']]
	df = df.rename(columns = {'DRUG_NAME': 'ID1', 'CELL_LINE_NAME': 'ID2', 'SMILES': 'X1', 'Expression': 'X2', 'LN_IC50': 'Y'})

	df.reset_index(drop = True).to_pickle('GSDC1.csv')