def HuRI_process():
	import pandas as pd
	# Download from HuRI
	df = pd.read_csv('/Users/kexinhuang/Downloads/HuRI.tsv', sep = '\t', header = None).rename(columns = {0: 'ID1', 1: 'ID2'})

	# process to line by line ensembl IDs and sent to biomart for batch query
	import numpy as np
	with open('./data/huri_protein.txt', 'w') as f:
	    for i in np.unique(np.array(df.ID1.values.tolist() + df.ID2.values.tolist())):
	        f.write(i)
	        f.write('\n')

	# select peptide output in the biomart and process
	import fastaparser
	IDs = []
	Seqs = []
	with open("/Users/kexinhuang/Downloads/mart_export.txt") as fasta_file:
	    parser = fastaparser.Reader(fasta_file)
	    for seq in parser:
	        # seq is a FastaSequence object
	        IDs.append(seq.id.split('|')[0])
	        Seqs.append(seq.sequence_as_string())


	df_t = pd.DataFrame(zip(IDs, Seqs)).drop_duplicates()
	df_t = df_t.rename(columns = {0: 'ID', 1: 'Seq'})
	df_t = df_t[df_t.Seq != 'SEQUENCE UNAVAILABLE']
	df_t = df_t.drop_duplicates()
	# When one gene id associated with multiple peptides, we concatenate them with separate sign *
	df_t = df_t.groupby('ID').Seq.apply(lambda x: ''.join(x))
	df_t = df_t.reset_index()
	id2seq = dict(zip(df_t['ID'].values, df_t['Seq'].values))
	# 8272 to 8248, filter out
	df = df[df.ID1.isin(list(id2seq.keys()))]
	df = df[df.ID2.isin(list(id2seq.keys()))]
	df['X1'] = [id2seq[i] for i in df.ID1.values]
	df['X2'] = [id2seq[i] for i in df.ID2.values]
	# 52369 interactions
	df.reset_index(drop = True).to_csv('./data/data_clean/interaction_prediction/PPI/HuRI.csv', index = False)