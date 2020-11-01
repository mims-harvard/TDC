def bepitope_iedb_process():

	from Bio import SeqIO
	import pandas as pd

	record_dict = SeqIO.to_dict(SeqIO.parse("/Users/kexinhuang/Downloads/antibody_data/BepiPred_epitope/BepiPred_pdb_chains.fasta", "fasta"))

	data = {}
	for i,j in record_dict.items():
	    s = str(j.seq)
	    y = [idx for idx, chr in enumerate(s) if chr.isupper()] 
	    data[s.upper()] = y

	df = pd.DataFrame(columns=['ID','X','Y'])

	for row, (k ,v) in enumerate(data.items()) :
	    df.loc[row, "ID"] = 'Protein ' + str(row + 1)
	    df.loc[row,"X"] = k
	    df.loc[row,"Y"] = v

	from ast import literal_eval
	df.Y = df.Y.apply(literal_eval)

def bepitope_pdb_process():

	from Bio import SeqIO
	import pandas as pd

	read_file = "/Users/kexinhuang/Downloads/antibody_data/BepiPred_epitope/BepiPred_pdb_chains.fasta"
	seq = []
	for record in SeqIO.parse(read_file, "fasta"):
	    seq.append(str(record.seq))
	    
	data = {}
	for s in seq:
	    y = [idx for idx, chr in enumerate(s) if chr.isupper()] 
	    data[s.upper()] = y

	df = pd.DataFrame(columns=['ID','X','Y'])

	for row, (k ,v) in enumerate(data.items()) :
	    df.loc[row, "ID"] = 'Protein ' + str(row + 1)
	    df.loc[row,"X"] = k
	    df.loc[row,"Y"] = v
	    
	from ast import literal_eval
	df.Y = df.Y.apply(literal_eval)