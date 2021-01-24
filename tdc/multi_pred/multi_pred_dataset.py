import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .. import base_dataset
from ..utils import *

class DataLoader(base_dataset.DataLoader):
	def __init__(self, name, path, print_stats, dataset_names):
		if name.lower() in dataset2target_lists.keys():
			if label_name is None:
				raise ValueError("Please select a label name. You can use tdc.utils.retrieve_label_name_list('" + name.lower() + "') to retrieve all available label names.")

		df = multi_dataset_load(name, path, dataset_names)

		self.df = df
		self.name = name
		self.path = path

	def get_data(self, format = 'df'):
		if format == 'df':
			return self.df
		elif format == 'dict':
			return dict(self.df)
		else:
			raise AttributeError("Please use the correct format input")

	def print_stats(self):
		print_sys('--- Dataset Statistics ---')
		print(str(len(self.df)) + ' data points.', flush = True, file = sys.stderr)
		print_sys('--------------------------')

	def get_split(self, method = 'random', seed = 42, frac = [0.7, 0.1, 0.2], column_name = None):

		df = self.get_data(format = 'df')

		if method == 'random':
			return create_fold(df, seed, frac)
		if (column_name is not None) and (column_name in df.columns.values): 
			if method == 'cold_split':		
				return create_fold_setting_cold(df, seed, frac, column_name)
		elif method == 'combination':
			return create_combination_split(df, seed, frac)
		else:
			raise AttributeError("Please select from random_split, or cold_split, if cold split. please specify the column name!")