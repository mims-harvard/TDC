import os, sys
import pandas as pd
from .label_name_list import dataset2target_lists

def get_label_map(name, path = './data', target = None, file_format = 'csv', output_format = 'dict', task = 'DDI'):
	name = fuzzy_search(name, dataset_names[task])
	if target is None:
		target = 'Y'
	df = pd_load(name, path)

	if output_format == 'dict':
		return dict(zip(df[target].values, df['Map'].values))
	elif output_format == 'df':
		return df
	elif output_format == 'array':
		return df['Map'].values
	else:
		raise ValueError("Please use the correct output format, select from dict, df, array.")

def get_reaction_type(name, path = './data', output_format = 'array'):
	name = fuzzy_search(name, dataset_names['RetroSyn'])
	df = pd_load(name, path)

	if output_format == 'df':
		return df
	elif output_format == 'array':
		return df['category'].values
	else:
		raise ValueError("Please use the correct output format, select from df, array.")

def retrieve_label_name_list(name):
	name = fuzzy_search(name, dataset_list)
	return dataset2target_lists[name]

def retrieve_dataset_names(name):
	return dataset_names[name]

def retrieve_all_benchmarks():
	return list(benchmark_names.keys())

def retrieve_benchmark_names(name):
	name = fuzzy_search(name, list(benchmark_names.keys()))
	datasets = benchmark_names[name]

	dataset_names = []

	for task, datasets in datasets.items():
		for dataset in datasets:
			dataset_names.append(dataset)
	return dataset_names