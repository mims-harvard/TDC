def TAP():
	import pandas as pd
	dev = pd.read_excel('/Users/kexinhuang/Downloads/pnas.1810576116.sd03.xlsx')
	seq = pd.read_excel('/Users/kexinhuang/Downloads/pnas.1810576116.sd02.xlsx')
	df = pd.DataFrame()
	df['X'] = [i for i in seq[['Heavy Sequence', 'Light Sequence']].values]
	df['Therapeutic'] = seq['Antibody Therapeutic'].values
	df.merge(dev, on = 'Therapeutic').rename(columns = {'Therapeutic': 'ID', 'Total CDR Length': 'CDR_Length', 'CDR Vic. PSH (Kyte & Doolittle)': 'PSH', 'CDR Vic. PPC': 'PPC', 'CDR Vic. PNC': 'PNC'}).to_csv('/Users/kexinhuang/Desktop/TAP.csv', index = False)


def sabdab_chen():
	from Bio import SeqIO
	import pandas as pd

	record_dict = SeqIO.to_dict(SeqIO.parse("/Users/kexinhuang/Downloads/sabdab_sequences_VL.fa", "fasta"))

	L_data = {}
	for i,j in record_dict.items():
	    s = str(j.seq)
	    L_data[i.split('|')[0].split('_')[0]] = s
	    
	record_dict = SeqIO.to_dict(SeqIO.parse("/Users/kexinhuang/Downloads/sabdab_sequences_VH.fa", "fasta"))

	H_data = {}
	for i,j in record_dict.items():
	    s = str(j.seq)
	    H_data[i.split('|')[0].split('_')[0]] = s
	df = pd.read_csv('/Users/kexinhuang/Downloads/DI_out.csv')
	df['Name'] = df.Name.apply(lambda x: x.split('_')[0])
	import numpy as np
	thr = np.quantile(df['Developability Index (Fv)'].values, 0.2)
	df = df.rename(columns = {'Name': 'ID', 'Developability Index (Fv)': 'Y'})
	df['Y'] = [1 if i < thr else 0 for i in df['Y'].values]

	df['X'] = df['ID'].apply(lambda x: [H_data[x], L_data[x]])
	df.to_csv('/Users/kexinhuang/Desktop/sabdab_chen.csv')