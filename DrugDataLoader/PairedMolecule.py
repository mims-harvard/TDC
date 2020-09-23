import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")
from DrugDataLoader import BaseDataset
from PairedMolecule_utils import * 
import utils 

class DataLoader(BaseDataset.DataLoader):
	'''
		DataLoader for paired molecules 
		it contains 
			(1) input molecule X 
			(2) target molecule Y
			where Y is similar molecule with more desirable molecular property, 
			e.g., QED score, DRD score
		
	'''
	def __init__(self, name, path = './data', print_stats = True):
		if not os.path.exists(path):
			os.makedirs(path)
		try:
			name = name.strip().lower()
			assert name in ['drd2', 'qed', 'logp']
			X_lst, Y_lst = eval(name + '_process(name, path)')
			self.X_lst = X_lst 
			self.Y_lst = Y_lst 
			self.name = name
		except:
			raise AttributeError("Please use the correct and available dataset name!")

	def get_data(self, format = 'df'):
		'''
			format: 

		'''

		if format == 'df':
			return pd.DataFrame({'X':self.X_lst, 'Y':self.Y_lst}) 
		elif format == 'dict':
			return {'X':self.X_lst, 'Y':self.Y_lst}
		else:
			return 

	def get_split(self, method = 'random', seed = 'benchmark', frac = [0.7, 0.1, 0.2]):
		'''
		Arguments:
			method: splitting schemes: random, cold_drug, scaffold split
			seed: 'benchmark' seed set to 1234, or int values
			frac: train/val/test split
		'''

		if seed == 'benchmark':
			seed = 1234

		df = self.get_data(format = 'df')
		
		if method == 'random':
			return utils.create_fold(df, seed, frac)
		else:
			raise AttributeError("Please specify the correct splitting method")



	def print_stats(self):
		print(self.name +  ' has ' + str(len(self.X_lst)) + ' molecule pairs.', flush = True, file = sys.stderr)




if __name__ == "__main__":
	dataloader = DataLoader(name = 'drd2', path = './data')
	dataloader.print_stats()
	pd_frame = dataloader.get_data()
	split_data = dataloader.get_split(method = 'random', seed = 'benchmark')
	print(split_data.keys())

	'''
	dataloader = DataLoader(name = 'qed', path = './data')
	pd_frame = dataloader.get_data()
	dataloader.print_stats()


	dataloader = DataLoader(name = 'logp', path = './data')
	pd_frame = dataloader.get_data()
	dataloader.print_stats()
	'''







