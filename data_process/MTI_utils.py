def miRTarBase_MTI():
	import pandas as pd
	df = pd.read_excel('/Users/kexinhuang/Downloads/miRTarBase_MTI.xlsx')
	with open('/Users/kexinhuang/Desktop/miRNA_names.txt', 'w') as f:
    for i in df['Target Gene (Entrez ID)'].unique():
        f.write(str(i))
        f.write('\n')

    from Bio import SeqIO
	import pandas as pd

	record_dict = SeqIO.to_dict(SeqIO.parse("/Users/kexinhuang/Downloads/mature.fa", "fasta"))

	mature_data = {}
	for i,j in record_dict.items():
	    s = str(j.seq)
	    mature_data[i.split('|')[0].split('_')[0]] = s
	import numpy as np
	inter = np.intersect1d(np.array(list(mature_data.keys())), df.miRNA.unique())

	df = df[['miRNA', 'Target Gene (Entrez ID)']]
	df = df.rename(columns = {'miRNA': 'ID1', 'Target Gene (Entrez ID)': 'ID2'})
	df_gene = pd.read_csv('/Users/kexinhuang/Downloads/uniprot-yourlist_M20201108A94466D2655679D1FD8953E075198DA80E25B26-filtered-rev-- (1).tab', sep = '\t')
	gene2seq = dict(df_gene[['yourlist:M20201108A94466D2655679D1FD8953E075198DA80E25B26', 'Sequence']].values)
	df['ID2'] = df['ID2'].apply(lambda x: str(x))
	df = df[df['ID2'].isin(gene2seq.keys())]
	df = df[df['ID1'].isin(mature_data.keys())]
	df['X1'] = [mature_data[i] for i in df['ID1']]
	df['X2'] = [gene2seq[i] for i in df['ID2']]
	df['Y'] = 1
	df.reset_index(drop = True).to_csv('/Users/kexinhuang/Desktop/mirtarbase.csv', index = False)