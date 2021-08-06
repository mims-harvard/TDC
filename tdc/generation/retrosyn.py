# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings
warnings.filterwarnings("ignore")

from . import generation_dataset
from ..metadata import dataset_names

class RetroSyn(generation_dataset.PairedDataLoader):

	"""Data loader class accessing to retro-synthetic prediction task.
	"""
	
	def __init__(self, name, path = './data', print_stats = False, input_name = 'product', output_name = 'reactant'): 
		"""To create an data loader object for forward reaction prediction task. The goal is to predict 
		the reaction products given a set of reactants
		
		Args:
		    name (str): the name of the datset
		    path (str, optional): the path to the saved data file.
		    print_stats (bool, optional): whether to print the basic statistics
		    input_name (str, optional): the name of the column containing input molecular data (product)
		    output_name (str, optional): the name of the column containing output molecular data (reactant)
		"""
		super().__init__(name, path, print_stats, input_name, output_name)