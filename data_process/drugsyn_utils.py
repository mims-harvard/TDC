def OncoPolyPharmacology_process():
	import numpy as np
	import pandas as pd
	import pickle 
	import gzip

	file = gzip.open('/Users/kexinhuang/Downloads/X.p.gz', 'rb')
	X = pickle.load(file)
	file.close()
	X = X[:23052]

	def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
	    if std1 is None:
	        std1 = np.nanstd(X, axis=0)
	    if feat_filt is None:
	        feat_filt = std1!=0
	    X = X[:,feat_filt]
	    X = np.ascontiguousarray(X)
	    if means1 is None:
	        means1 = np.mean(X, axis=0)
	    X = (X-means1)/std1[feat_filt]
	    if norm == 'norm':
	        return(X, means1, std1, feat_filt)
	    elif norm == 'tanh':
	        return(np.tanh(X), means1, std1, feat_filt)
	    elif norm == 'tanh_norm':
	        X = np.tanh(X)
	        if means2 is None:
	            means2 = np.mean(X, axis=0)
	        if std2 is None:
	            std2 = np.std(X, axis=0)
	        X = (X-means2)/std2
	        X[:,std2==0]=0
	    return(X, means1, std1, means2, std2, feat_filt)
	X = X[0]
	df = pd.read_csv('/Users/kexinhuang/Downloads/merck/labels.csv')
	df['cell line'] = [i for i in X]
	df_smiles = pd.read_csv('/Users/kexinhuang/Downloads/merck/smiles.csv', header = None)
	drug2smiles = dict(df_smiles.values)

	df['X1'] = [drug2smiles[i] for i in df.drug_a_name]
	df['X2'] = [drug2smiles[i] for i in df.drug_b_name]
	df = df.drop(['Unnamed: 0', 'fold'], axis = 1)
	df.rename(columns = {'drug_a_name': 'Drug1_ID', 'drug_b_name': 'Drug2_ID', 'cell line': 'Cell_Line', 'cell_line': 'Cell_Line_ID', 'X1': 'Drug1', 'X2': 'Drug2', 'synergy': 'Y'})

X = normalize(X)

def NCI_60_process():
	import pandas as pd
	df = pd.read_csv('/Users/kexinhuang/Downloads/comboFM_data/NCI-ALMANAC_full_data.csv')
	cell_lines_exp = pd.read_csv('/Users/kexinhuang/Downloads/comboFM_data/additional_data/NCI-60__gene_expression.txt', sep = ' ')
	columns = [i.split(':')[1] for i in cell_lines_exp.drop(['Unnamed: 0'], axis = 1).columns.values]
	cell2exp = dict(zip(columns, cell_lines_exp.drop(['Unnamed: 0'], axis = 1).T.values))