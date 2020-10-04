import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .. import base_dataset
from ..utils import *

class DataLoader(base_dataset.DataLoader):
	def __init__(self, name, path = './data', target = None, print_stats = True, dataset_names = dataset_names):
		if name in dataset_names: 
			entity1, y, entity1_idx = dataset_load(name, path, target, 'csv')
		else:
			raise AttributeError("Dataset does not exist. Please use the correct and available dataset!")
		
		self.entity1 = entity1
		self.y = y
		self.entity1_idx = entity1_idx
		self.name = name
		self.entity1_name = 'Drug'
		self.path = path
		self.file_format = 'csv'
		self.target = target
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

	def get_data(self, format = 'df'):
		'''
		Arguments:
			df: return pandas DataFrame; if not true, return np.arrays			
		returns:
			self.drugs: drug smiles strings np.array
			self.targets: target Amino Acid Sequence np.array
			self.y: inter   action score np.array
		'''
		if format == 'df':
			return pd.DataFrame({self.entity1_name + '_ID': self.entity1_idx, self.entity1_name: self.entity1, 'Y': self.y})
		elif format == 'dict':
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

		df = self.get_data(format = 'df')

		if method == 'random':
			return create_fold(df, seed, frac)
		elif method == 'cold_' + self.entity1_name.lower():
			return create_fold_setting_cold(df, seed, frac, self.entity1_name)
		elif method == 'scaffold':
			return create_scaffold_split(df, seed, frac, self.entity1_name)
		else:
			raise AttributeError("Please specify the correct splitting method")

	def print_stats(self):
		print('There are ' + str(len(np.unique(self.entity1))) + ' unique ' + self.entity1_name.lower() + 's', flush = True, file = sys.stderr)