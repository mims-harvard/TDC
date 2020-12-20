import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .. import base_dataset
from ..utils import *

class DataLoader(base_dataset.DataLoader):
	def __init__(self, name, path, label_name, print_stats, dataset_names):
		if name.lower() in dataset2target_lists.keys():
			#print_sys("Tip: Use tdc.utils.retrieve_label_name_list('" + name.lower() + "') to retrieve all available label names.")
			if label_name is None:
				raise ValueError("Please select a label name. You can use tdc.utils.retrieve_label_name_list('" + name.lower() + "') to retrieve all available label names.")

		entity1, y, entity1_idx = property_dataset_load(name, path, label_name, dataset_names)
		
		self.entity1 = entity1
		self.y = y
		self.entity1_idx = entity1_idx
		self.name = name
		self.entity1_name = 'Drug'
		self.path = path
		self.file_format = 'csv'
		self.label_name = label_name

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
			return {self.entity1_name + '_ID': self.entity1_idx.values, self.entity1_name: self.entity1.values, 'Y': self.y.values}
		elif format == 'DeepPurpose':
			return self.entity1.values, self.y.values
		elif format == 'sklearn':
			pass
		else:
			raise AttributeError("Please use the correct format input")

	def get_split(self, method = 'random', seed = 42, frac = [0.7, 0.1, 0.2]):
		'''
		Arguments:
			method: splitting schemes: random, cold_drug, scaffold split
			seed: default 42
			frac: train/val/test split
		'''

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
		print_sys('--- Dataset Statistics ---')
		try:
			x = np.unique(self.entity1)
		except:
			x = np.unique(self.entity1_idx)

		print(str(len(x)) + ' unique ' + self.entity1_name.lower() + 's.', flush = True, file = sys.stderr)
		print_sys('--------------------------')
