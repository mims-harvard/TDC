import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .. import BaseDataset, utils
from .MolGenPaired_utils import * 
from .Reaction_utils import * 

class MolGenPaired(BaseDataset.DataLoader):
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

class Reaction(BaseDataset.DataLoader):
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
			assert name in ['uspto']
			reactant_lst, catalyst_lst, product_lst  = eval(name + '_process(name, path)')
			self.reactant_lst = reactant_lst 
			self.catalyst_lst = catalyst_lst
			self.product_lst = product_lst
			self.name = name
		except:
			raise AttributeError("Please use the correct and available dataset name!")

	def get_data(self, format = 'df'):
		'''
			format: 

		'''

		if format == 'df':
			return pd.DataFrame({'reactant':self.reactant_lst, 'product':self.product_lst, 'catalyst':self.catalyst_lst}) 
		elif format == 'dict':
			return {'reactant':self.reactant_lst, 'product':self.product_lst, 'catalyst':self.catalyst_lst}
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

	def catalyst_statistics(self):
		'''
			compute some statistics for catalyst, e.g., count/frequency, total num. 


				('', 418450)
				('C1COCC1', 98588)
				('C(Cl)Cl', 91790)
				('CN(C=O)C', 77304)
				('ClCCl', 64883)
				('CO', 64824)
				('O', 49760)
				('O1CCCC1', 45353)
				('C(O)C', 41220)
				('CN(C)C=O', 32733)
				('C(#N)C', 29184)
				('C1(C)C=CC=CC=1', 26739)
				('O1CCOCC1', 25727)
				('C(OCC)(=O)C', 16592)
				('CCO', 15147)
				('C(O)(=O)C', 13912)
				('CS(C)=O', 13893)
				('N1C=CC=CC=1', 13378)
				('CN(C=O)C.O', 13325)
				('CO.[Pd]', 13007)
		'''
		catalyst_num = len(set(self.catalyst_lst))
		print(str(catalyst_num) + " catalysts totally.")
		## 55595 catalysts totally. 
		from collections import defaultdict, Counter
		catalyst2cnt = defaultdict(lambda:0)
		for i in self.catalyst_lst:
			catalyst2cnt[i]+=1 
		for i in sorted(catalyst2cnt.items(), key=lambda x:x[1], reverse = True)[:20]:
			print(i)
		return 


	def print_stats(self):
		print(self.name +  ' has ' + str(len(self.reactant_lst)) + ' reactions.', flush = True, file = sys.stderr)






