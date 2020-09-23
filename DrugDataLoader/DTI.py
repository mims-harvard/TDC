import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from DrugDataLoader import utils, Interaction

class DataLoader(Interaction.DataLoader):
	def __init__(self, name, path = './data', target = None, print_stats = True):
		super().__init__(name, path, target, print_stats)
		self.entity1_name = 'Drug'
		self.entity2_name = 'Target'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)