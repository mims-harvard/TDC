import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .. import base_dataset
from ..utils import *


class DataLoader(base_dataset.DataLoader):
	def __init__(self, name, path, target, print_stats, dataset_names):
		'''
		Arguments:
			name: fuzzy name of the generation dataset. e.g., uspto50k, qed, drd, ... 
			path: directory path that stores the dataset, e.g., ./data
			print_stats: bool, whether print the stats.  
			dataset_names: exact names of dataset  e.g., 
		returns:
			None
		'''
		pass 

	# def __init__(self, name, path, print_stats, dataset_names):
	# 	if name.lower() in retrosyn_dataset_names.keys():  
	# 		print_sys("Tip: Use tdc.utils.target_list('" + name.lower() + "') to retrieve all available label targets.")


