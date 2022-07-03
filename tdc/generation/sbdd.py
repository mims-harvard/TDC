# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT"

import warnings
warnings.filterwarnings("ignore")

from . import bi_generation_dataset
from ..metadata import dataset_names

class SBDD(bi_generation_dataset.DataLoader):

	"""Data loader class accessing to pocket-based ligand generation task.
	"""
	
	def __init__(self, name, path = './data', print_stats = False, return_pocket=False, threshold=15, remove_Hs=True, keep_het=False, allowed_atom_list = ['C', 'N', 'O', 'S', 'H', 'B', 'Br', 'Cl', 'P', 'I', 'F']): 
		"""To create an data loader object for pocket-based ligand generation task. The goal is to generate ligands 
		that bind to a given protein pocket. 
		
		Args:
		    name (str): the name of the datset
		    path (str, optional): the path to the saved data file.
		    print_stats (bool, optional): whether to print the basic statistics
		"""
		super().__init__(name, path, print_stats, return_pocket, threshold, remove_Hs, keep_het, allowed_atom_list)