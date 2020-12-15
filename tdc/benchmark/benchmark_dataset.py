import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from ..utils import *

class DataLoader:
	def __init__(self, name, path):
		name = fuzzy_search(name, retrieve_all_benchmarks())
		self.datasets = retrieve_benchmark_names(name)

	def get_data(self):
		pass

	def get_cross_validation(self):
		pass
		
	def evaluate(self, pred):
		raise NotImplementedError



from tdc import benchmark
'''
SAVE function?

-- PATH
	-- ADMET_Benchmark
		-- HIA_Hou
			-- train.csv
			-- valid.csv
			-- test.csv
		-- Caco2_Wang
			-- train.csv
			-- valid.csv
			-- test.csv
		....

Should we do just one split? or like five splits including test set? if we fix one testing set, should we provide fixed cross-validation as well? would that be too much? as we are only reporting test metric

'''
bm = benchmark(name = 'ADMET', path = 'data/')
bm.get_local_copy() # get a local copy

for dataset in bm.data_iter():
	train, valid, test = dataset.get_split()

	### -- Model -- ###