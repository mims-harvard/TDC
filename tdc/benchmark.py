import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .utils import *
from .metadata import get_task2category, bm_metric_names, benchmark_names, bm_split_names
from .evaluator import Evaluator

class BenchmarkGroup:
	def __init__(self, name, path = './data', file_format='csv'):
		'''
		-- PATH
			-- ADMET_Benchmark
				-- HIA_Hou
					-- train_val.csv
					-- test.csv
				-- Caco2_Wang
					-- train_val.csv
					-- test.csv
				....
		from tdc import BenchmarkGroup
		group = BenchmarkGroup(name = 'ADMET_Group', path = 'data/')
		predictions = {}

		for benchmark in group:
		   name = benchmark['name']
		   train_val, test = benchmark['train_val'], benchmark['test']
		   
		   # to obtain any number of train:valid split
		   train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = 42, frac = [0.875, 0.125])

		   ## --- train your model --- ##
		   predictions[name] = y_pred

		group.evaluate(predictions)
		# {'caco2_wang': 0.234, 'hia_hou': 0.786}

		benchmark = group.get('Caco2_Wang')
		train, valid, test = benchmark['train'], benchmark['valid'], benchmark['test']
		## --- train your model --- ##
		group.evaluate(y_pred, benchmark = 'Caco2_Wang')
		# 0.234
		
		from tdc import BenchmarkGroup
		group = BenchmarkGroup(name = 'ADMET_Group', path = 'data/')
		predictions_list = []

		for seed in [1, 2, 3, 4, 5]:
		    predictions = {}
		    for benchmark in group:
		        name = benchmark['name']
		        train_val, test = benchmark['train_val'], benchmark['test']
		        train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
		        ## --- train your model --- ##
		        y_pred = [1] * len(test)
		        predictions[name] = y_pred
		    predictions_list.append(predictions)

		group.evaluate_many(predictions_list)

		'''
		
		self.name = bm_group_load(name, path)
		self.path = os.path.join(path, self.name)
		self.datasets = benchmark_names[self.name]
		self.dataset_names = []
		self.file_format = file_format

		for task, datasets in self.datasets.items():
			for dataset in datasets:
				self.dataset_names.append(dataset)


	def __iter__(self):
		self.index = 0
		self.num_datasets = len(self.dataset_names)
		return self

	def __next__(self):
		if self.index < self.num_datasets:
			dataset = self.dataset_names[self.index]
			print_sys('--- ' + dataset + ' ---')
			data_path = os.path.join(self.path, dataset)
			if self.file_format == 'csv':
				train = pd.read_csv(os.path.join(data_path, 'train_val.csv'))
				test = pd.read_csv(os.path.join(data_path, 'test.csv'))
			elif self.file_format == 'pkl':
				train = pd.read_pickle(os.path.join(data_path, 'train_val.pkl'))
				test = pd.read_pickle(os.path.join(data_path, 'test.pkl'))
			self.index += 1
			return {'train_val': train, 'test': test, 'name': dataset}
		else:
			raise StopIteration
			
	def get_train_valid_split(self, seed, benchmark, split_type = 'default'):
		print_sys('generating training, validation splits...')
		dataset = fuzzy_search(benchmark, self.dataset_names)
		data_path = os.path.join(self.path, dataset)
		if self.file_format == 'csv':
			train_val = pd.read_csv(os.path.join(data_path, 'train_val.csv'))
		elif self.file_format == 'pkl':
			train_val = pd.read_pickle(os.path.join(data_path, 'train_val.pkl'))

		if split_type == 'default':
			split_method = bm_split_names[self.name][dataset]
		else:
			split_method = split_type

		frac = [0.875, 0.125, 0.0]
		'''
		if len(frac) == 3:
			# train:val:test split
			train_frac = frac[0]/(frac[0] + frac[1])
			valid_frac = 1 - train_frac
			frac = [train_frac, valid_frac, 0.0]
		else:
			# train:val split
			frac = [frac[0], frac[1], 0.0]
		'''
		if split_method == 'scaffold':
			out = create_scaffold_split(train_val, seed, frac = frac, entity = 'Drug')
		elif split_method == 'random':
			out = create_fold(train_val, seed, frac = frac)
		elif split_method == 'combination':
			out = create_combination_split(train_val, seed, frac=frac)
		else:
			raise NotImplementedError
		return out['train'], out['valid']

	def get(self, benchmark):
		dataset = fuzzy_search(benchmark, self.dataset_names)
		data_path = os.path.join(self.path, dataset)
		if self.file_format == 'csv':
			train = pd.read_csv(os.path.join(data_path, 'train_val.csv'))
			test = pd.read_csv(os.path.join(data_path, 'test.csv'))
		elif self.file_format == 'pkl':
			train = pd.read_pickle(os.path.join(data_path, 'train_val.pkl'))
			test = pd.read_pickle(os.path.join(data_path, 'test.pkl'))
		return {'train': train, 'test': test, 'name': dataset}

	def evaluate(self, pred, true = None, benchmark = None):
		if true is None:
			# test set evaluation
			metric_dict = bm_metric_names[self.name]
			out = {}
			for data_name, pred_ in pred.items():
				data_name = fuzzy_search(data_name, self.dataset_names)
				data_path = os.path.join(self.path, data_name)
				if self.file_format == 'csv':
					test = pd.read_csv(os.path.join(data_path, 'test.csv'))
				elif self.file_format == 'pkl':
					test = pd.read_pickle(os.path.join(data_path, 'test.pkl'))
				y = test.Y.values
				evaluator = eval('Evaluator(name = \'' + metric_dict[data_name] + '\')')
				out[data_name] = {metric_dict[data_name]: round(evaluator(y, pred_), 3)}

				# If reporting accuracy across target classes
				if 'target_class' in test.columns:
					test['pred'] = pred_
					for c in test['target_class'].unique():
						data_name_subset = data_name + '_' + c
						test_subset = test[test['target_class']==c]
						y_subset = test_subset.Y.values
						pred_subset = test_subset.pred.values

						evaluator = eval('Evaluator(name = \'' +
									     metric_dict[data_name_subset] + '\')')
						out[data_name_subset] = {metric_dict[data_name_subset]:
							        round(evaluator(y_subset, pred_subset), 3)}
			return out
		else:
			# validation set evaluation
			if benchmark is None:
				raise ValueError('Please specify the benchmark name for us to retrieve the standard metric!')
			data_name = fuzzy_search(benchmark, self.dataset_names)
			metric_dict = bm_metric_names[self.name]
			evaluator = eval('Evaluator(name = \'' + metric_dict[data_name] + '\')')
			return {metric_dict[data_name]: round(evaluator(true, pred), 3)}

	def evaluate_many(self, preds):
		"""
		:param preds: list of dict<str dataset_name: list of float>
		:return: dict<dataset_name: [mean_metric_result, std_metric_result]

		This function returns the data in a format needed to submit to the Leaderboard
		"""
		if len(preds) < 5:
			return ValueError("Must have predictions from at least five runs for leaderboard submission")
		individual_results = []
		for pred in preds:
			retval = self.evaluate(pred)
			individual_results.append(retval)

		given_dataset_names = list(individual_results[0].keys())
		aggregated_results = {}
		for dataset_name in given_dataset_names:
			my_results = []
			for individual_result in individual_results:
				my_result = list(individual_result[dataset_name].values())[0]
				my_results.append(my_result)
			u = np.mean(my_results)
			std = np.std(my_results)
			aggregated_results[dataset_name] = [round(u, 3), round(std, 3)]
		return aggregated_results
