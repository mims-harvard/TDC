def ProteinAntigen_SAbDab():
	import pandas as pd
	# from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search?ABtype=All&method=All&species=All&resolution=&rfactor=&antigen=All&ltype=All&constantregion=All&affinity=True&chothiapos=&restype=ALA
	df = pd.read_csv('/Users/kexinhuang/Downloads/antibody_data/sab.tsv', sep = '\t')
	df = df[~df.antigen_chain.isnull()]
	df = df[df.antigen_type.isin(['protein', 'peptide'])]
	df = df.drop_duplicates('pdb')
	df = df[['pdb', 'Hchain', 'Lchain', 'antigen_chain', 'antigen_name', 'affinity']]
	df = df[df.Lchain.notnull()]
	df = df[df.Hchain.notnull()]
	df = df[df.antigen_chain.notnull()]
	import urllib.request

	from tqdm import tqdm
	seq_all = {}
	for i in tqdm(range(len(df))):
	    try:
	        seqs = {}
	        with urllib.request.urlopen('https://www.rcsb.org/fasta/entry/' + df.iloc[i].pdb.upper() + '/download') as f:
	            html = f.read().decode('utf-8')
	        
	        chain2seq = dict(zip([x.split('|')[1].split(' ')[1] for x in html.split('>')[1:]], [j.split('\n')[1] for j in html.split('>')[1:]]))
	        keys = list(chain2seq.keys())
	        for idx in keys:
	            if len(idx.split(',')) > 1:
	                for x in idx.split(','):
	                    chain2seq[x] = chain2seq[idx]
	        seqs['Hchain'] = chain2seq[df.iloc[i].Hchain]
	        seqs['Lchain'] = chain2seq[df.iloc[i].Lchain]
	        seqs['antigen_chain'] = chain2seq[df.iloc[i].antigen_chain]
	        seq_all[df.iloc[i].pdb] = seqs
	    except:
	        print(i)
	df_out = pd.DataFrame(seq_all).T.reset_index().rename(columns = {'index': 'pdb', 'Hchain': 'H_seq', 'Lchain': 'L_seq', 'antigen_chain': 'antigen_seq'}).merge(df, on = 'pdb', how = 'left')
	df_out.to_csv('ProteinAntigen_SAbDab.csv', index = False)