import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from DrugDataLoader import utils, DrugProperty
from DrugDataLoader.QM_utils import *

class DataLoader(DrugProperty.DataLoader):
	def __init__(self, name, path = './data', target = None, print_stats = True):
		try:
			entity1, y, entity1_idx = eval(name + '_process(name, path, target)')

			self.entity1 = entity1
			self.y = y
			self.entity1_idx = entity1_idx
			self.name = name
		except:
			raise AttributeError("Please use the correct and available dataset name!")

		self.entity1_name = 'Drug'

		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

	def get_data(self, format = 'dict'):
		'''
		Arguments:
			df: return pandas DataFrame; if not true, return np.arrays			
		returns:
			self.drugs: drug smiles strings np.array
			self.targets: target Amino Acid Sequence np.array
			self.y: inter   action score np.array
		'''
		if format == 'df':
			utils.print_sys('the features are 2D distance map, thus is not suitable for pandas, switch to dictionary automatically...')
			format = 'dict'
		
		if format == 'dict':
			return {self.entity1_name + '_ID': self.entity1_idx, self.entity1_name: self.entity1, 'Y': self.y}
		elif format == 'DeepPurpose':
			return self.entity1, self.y
		elif format == 'sklearn':
			pass
		else:
			raise AttributeError("Please use the correct format input")

	def get_split(self, method = 'random', seed = 'benchmark', frac = [0.7, 0.1, 0.2]):
		'''
		Arguments:
			method: splitting schemes: random, cold_drug, scaffold split
			seed: 'benchmark' seed set to 1234, or int values
			frac: train/val/test split
		'''
		if seed == 'benchmark':
			seed = 1234

		df = self.get_data()

		if method == 'cold_' + self.entity1_name.lower():
			utils.print_sys("cold drug is the same as random split for drug property prediction...")
			method = 'random'

		if method == 'random':
			len_data = len(df['Drug_ID'])
			train, val, test = utils.train_val_test_split(len_data, frac, seed)
			return {'X_train': df['Drug'][train], 'y_train': df['Y'][train], 'X_val': df['Drug'][val], 'y_val': df['Y'][val], 'X_test': df['Drug'][test], 'y_test': df['Y'][test]}
		elif method == 'scaffold':
			raise AttributeError("Scaffold does not apply for QM dataset since the input features are 2D distance map")
		else:
			raise AttributeError("Please specify the correct splitting method")

	def print_stats(self):
		print('There are ' + str(self.entity1.shape[0]) + ' unique ' + self.entity1_name.lower() + 's', flush = True, file = sys.stderr)