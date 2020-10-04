import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from ..utils import *
from . import interaction_dataset

class DTI(interaction_dataset.DataLoader):
	def __init__(self, name, path = './data', target = None, print_stats = True):
		super().__init__(name, path, target, print_stats, dataset_names = dti_dataset_names)
		self.entity1_name = 'Drug'
		self.entity2_name = 'Target'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

class DDI(interaction_dataset.DataLoader):
	def __init__(self, name, path = './data', target = None, print_stats = True):
		super().__init__(name, path, target, print_stats, dataset_names = ddi_dataset_names)
		self.entity1_name = 'Drug1'
		self.entity2_name = 'Drug2'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

	def print_stats(self):
		print('There are ' + str(len(np.unique(self.entity1.tolist() + self.entity2.tolist()))) + ' unique drugs', flush = True, file = sys.stderr)
		print('There are ' + str(len(self.y)) + ' drug-drug pairs', flush = True, file = sys.stderr)

class PPI(interaction_dataset.DataLoader):
	def __init__(self, name, path = './data', target = None, print_stats = True):
		super().__init__(name, path, target, print_stats, dataset_names = ppi_dataset_names)
		self.entity1_name = 'Protein1'
		self.entity2_name = 'Protein2'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

	def print_stats(self):
		print('There are ' + str(len(np.unique(self.entity1.tolist() + self.entity2.tolist()))) + ' unique proteins', flush = True, file = sys.stderr)
		print('There are ' + str(len(self.y)) + ' protein-protein pairs', flush = True, file = sys.stderr)
