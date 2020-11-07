def TAP():
	import pandas as pd
	dev = pd.read_excel('/Users/kexinhuang/Downloads/pnas.1810576116.sd03.xlsx')
	seq = pd.read_excel('/Users/kexinhuang/Downloads/pnas.1810576116.sd02.xlsx')
	df = pd.DataFrame()
	df['X'] = [i for i in seq[['Heavy Sequence', 'Light Sequence']].values]
	df['Therapeutic'] = seq['Antibody Therapeutic'].values
	df.merge(dev, on = 'Therapeutic').rename(columns = {'Therapeutic': 'ID', 'Total CDR Length': 'CDR_Length', 'CDR Vic. PSH (Kyte & Doolittle)': 'PSH', 'CDR Vic. PPC': 'PPC', 'CDR Vic. PNC': 'PNC'}).to_csv('/Users/kexinhuang/Desktop/TAP.csv', index = False)