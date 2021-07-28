import os, sys
import numpy as np
import pandas as pd
import subprocess
import pickle
from fuzzywuzzy import fuzz

def fuzzy_search(name, dataset_names):
	name = name.lower()
	if name[:4] == 'tdc.':
		name = name[4:]
	if name in dataset_names:
		s =  name
	else:
		# print("========fuzzysearch=======", dataset_names, name)
		s =  get_closet_match(dataset_names, name)[0]
	if s in dataset_names:
		return s
	else:
		raise ValueError(s + " does not belong to this task, please refer to the correct task name!")

def get_closet_match(predefined_tokens, test_token, threshold=0.8):
	"""Get the closest match by Levenshtein Distance.

	Parameters
	----------
	predefined_tokens : list of string
		Predefined string tokens.

	test_token : string
		User input that needs matching to existing tokens.

	threshold : float in (0, 1), optional (default=0.8)
		The lowest match score to raise errors.

	Returns
	-------

	"""
	prob_list = []

	for token in predefined_tokens:
		# print(token)
		prob_list.append(
			fuzz.ratio(str(token).lower(), str(test_token).lower()))

	assert (len(prob_list) == len(predefined_tokens))

	prob_max = np.nanmax(prob_list)
	token_max = predefined_tokens[np.nanargmax(prob_list)]

	# match similarity is low
	if prob_max / 100 < threshold:
		print_sys(predefined_tokens)
		raise ValueError(test_token,
						 "does not match to available values. "
						 "Please double check.")
	return token_max, prob_max / 100

def save_dict(path, obj):
	with open(path, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(path):
	with open(path, 'rb') as f:
		return pickle.load(f)


def install(package):
	subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def print_sys(s):
	print(s, flush = True, file = sys.stderr)

def to_submission_format(results):
	df = pd.DataFrame(results)
	def get_metric(x):
		metric = []
		for i in x:
			metric.append(list(i.values())[0])
		return [round(np.mean(metric), 3), round(np.std(metric), 3)]
	return dict(df.apply(get_metric, axis = 1))
