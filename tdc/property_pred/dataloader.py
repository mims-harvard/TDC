"""Summary
"""
import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from . import property_dataset
from ..utils import *

class ADME(property_dataset.DataLoader):

	"""Summary
	"""
	
	def __init__(self, name, path = './data', target = None, print_stats = True):
		"""Summary
		
		Parameters
		----------
		name : TYPE
		    Description
		path : str, optional
		    Description
		target : None, optional
		    Description
		print_stats : bool, optional
		    Description
		"""
		super().__init__(name, path, target, dataset_names = adme_dataset_names)

class Toxicity(property_dataset.DataLoader):

	"""Summary
	"""
	
	def __init__(self, name, path = './data', target = None, print_stats = True):
		"""Summary
		
		Parameters
		----------
		name : TYPE
		    Description
		path : str, optional
		    Description
		target : None, optional
		    Description
		print_stats : bool, optional
		    Description
		"""
		super().__init__(name, path, target, dataset_names = toxicity_dataset_names)

class HTS(property_dataset.DataLoader):

	"""Summary
	"""
	
	def __init__(self, name, path = './data', target = None, print_stats = True):
		"""Summary
		
		Parameters
		----------
		name : TYPE
		    Description
		path : str, optional
		    Description
		target : None, optional
		    Description
		print_stats : bool, optional
		    Description
		"""
		super().__init__(name, path, target, dataset_names = hts_dataset_names)

class QM(property_dataset.DataLoader):

	"""Summary
	
	Attributes
	----------
	entity1 : TYPE
	    Description
	entity1_idx : TYPE
	    Description
	entity1_name : str
	    Description
	name : TYPE
	    Description
	y : TYPE
	    Description
	"""
	
	def __init__(self, name, path = './data', target = None, print_stats = True):
		"""Summary
		
		Parameters
		----------
		name : TYPE
		    Description
		path : str, optional
		    Description
		target : None, optional
		    Description
		print_stats : bool, optional
		    Description
		
		Raises
		------
		AttributeError
		    Description
		"""
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
		"""Summary
		
		Parameters
		----------
		format : str, optional
		    Description
		
		Returns
		-------
		TYPE
		    Description
		
		Raises
		------
		AttributeError
		    Description
		"""
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
		"""Summary
		
		Parameters
		----------
		method : str, optional
		    Description
		seed : str, optional
		    Description
		frac : list, optional
		    Description
		
		Returns
		-------
		TYPE
		    Description
		
		Raises
		------
		AttributeError
		    Description
		"""
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
		"""Summary
		"""
		print('There are ' + str(self.entity1.shape[0]) + ' unique ' + self.entity1_name.lower() + 's', flush = True, file = sys.stderr)