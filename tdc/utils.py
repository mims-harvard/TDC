import requests
from zipfile import ZipFile 
import os, sys
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import json
import warnings
warnings.filterwarnings("ignore")
import subprocess
import pickle
from fuzzywuzzy import fuzz
from tqdm import tqdm
from .metadata import name2type, name2id, dataset_list, dataset_names, benchmark_names, benchmark2id, benchmark2type
from .metadata import property_names, paired_dataset_names, single_molecule_dataset_names
from .metadata import retrosyn_dataset_names, forwardsyn_dataset_names, molgenpaired_dataset_names, generation_datasets
from .metadata import oracle2id, download_oracle_names, trivial_oracle_names, oracle_names, oracle2type 

from .label_name_list import dataset2target_lists

try:
	from urllib.error import HTTPError
	from urllib.parse import quote, urlencode
	from urllib.request import urlopen
except ImportError:
	from urllib import urlencode
	from urllib2 import quote, urlopen, HTTPError

def fuzzy_search(name, dataset_names):
	name = name.lower()
	if name in dataset_names:
		s =  name
	else:
		# print("========fuzzysearch=======", dataset_names, name)
		s =  get_closet_match(dataset_names, name)[0]
	if s in dataset_names:
		return s
	else:
		raise ValueError(s + " does not belong to this task, please refer to the correct task name!")

def download_wrapper(name, path, dataset_names):
	name = fuzzy_search(name, dataset_names)
	server_path = 'https://dataverse.harvard.edu/api/access/datafile/'

	dataset_path = server_path + str(name2id[name])

	if not os.path.exists(path):
		os.mkdir(path)

	if os.path.exists(os.path.join(path, name + '.' + name2type[name])):
		print_sys('Found local copy...')
	else:
		print_sys("Downloading...")
		dataverse_download(dataset_path, path, name, name2type)
	return name

def oracle_download_wrapper(name, path, oracle_names):
	name = fuzzy_search(name, oracle_names)
	if name in trivial_oracle_names:
		return name

	server_path = 'https://dataverse.harvard.edu/api/access/datafile/'
	dataset_path = server_path + str(oracle2id[name])

	if not os.path.exists(path):
		os.mkdir(path)

	if os.path.exists(os.path.join(path, name + '.' + oracle2type[name])):
		print_sys('Found local copy...')
	else:
		print_sys("Downloading Oracle...")
		dataverse_download(dataset_path, path, name, oracle2type) ## to-do to-check
		print_sys("Done!")
	return name

def bm_download_wrapper(name, path):
	name = fuzzy_search(name, list(benchmark_names.keys()))
	server_path = 'https://dataverse.harvard.edu/api/access/datafile/'
	dataset_path = server_path + str(benchmark2id[name])

	if not os.path.exists(path):
		os.mkdir(path)

	if os.path.exists(os.path.join(path, name)):
		print_sys('Found local copy...')
	else:
		print_sys('Downloading Benchmark Group...')
		dataverse_download(dataset_path, path, name, benchmark2type)
		print_sys('Extracting zip file...')
		with ZipFile(os.path.join(path, name + '.zip'), 'r') as zip:
			zip.extractall(path = os.path.join(path))
		print_sys("Done!")
	return name

def pd_load(name, path):
	try:
		if name2type[name] == 'tab':
			df = pd.read_csv(os.path.join(path, name + '.' + name2type[name]), sep = '\t')
		elif name2type[name] == 'csv':
			df = pd.read_csv(os.path.join(path, name + '.' + name2type[name]))
		elif name2type[name] == 'pkl':
			df = pd.read_pickle(os.path.join(path, name + '.' + name2type[name]))
		else:
			raise ValueError("The file type must be one of tab/csv/pickle.")
		try:
			df = df.drop_duplicates()
		except:
			pass
		return df
	except (EmptyDataError, EOFError) as e:
		import sys
		sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")

def property_dataset_load(name, path, target, dataset_names):
	if target is None:
		target = 'Y'
	name = download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	try:
		if target is not None:
			target = fuzzy_search(target, df.columns.values)
		df = df[df[target].notnull()].reset_index(drop = True)
	except:
		with open(os.path.join(path, name + '.' + name2type[name]), 'r') as f:
			flag = 'Service Unavailable' in ' '.join(f.readlines())
			if flag:
				import sys
				sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")
			else:
				sys.exit("Please report this error to cosamhkx@gmail.com, thanks!")
	try:
		return df['X'], df[target], df['ID']
	except:
		return df['Drug'], df[target], df['Drug_ID']

def molpair_process(name, path, dataset_names):
	name = download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	return df['input'], df['output']

def interaction_dataset_load(name, path, target, dataset_names):
	name = download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	try:
		if target is None:
			target = 'Y'
		if target not in df.columns.values:
			# for binary interaction data, the labels are all 1. negative samples can be sampled from utils.NegSample function
			df[target] = 1
		if target is not None:
			target = fuzzy_search(target, df.columns.values)
		df = df[df[target].notnull()].reset_index(drop = True)
		return df['X1'], df['X2'], df[target], df['ID1'], df['ID2']
	except:
		with open(os.path.join(path, name + '.' + name2type[name]), 'r') as f:
			flag = 'Service Unavailable' in ' '.join(f.readlines())
			if flag:
				import sys
				sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")
			else:
				sys.exit("Please report this error to cosamhkx@gmail.com, thanks!")


def multi_dataset_load(name, path, dataset_names):
	name = download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	return df

def generation_paired_dataset_load(name, path, dataset_names, input_name, output_name):
	name = download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	return df[input_name], df[output_name]


def distribution_dataset_load(name, path, dataset_names, column_name):
	name = download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	return df[column_name]

def generation_dataset_load(name, path, dataset_names):
	name = download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	return df['input'], df['target']

def oracle_load(name, path = './oracle', oracle_names = oracle_names):
	name = oracle_download_wrapper(name, path, oracle_names)
	return name

def bm_group_load(name, path):
	name = bm_download_wrapper(name, path)
	return name

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

def dataverse_download(url, path, name, types):
	save_path = os.path.join(path, name + '.' + types[name])
	response = requests.get(url, stream=True)
	total_size_in_bytes= int(response.headers.get('content-length', 0))
	block_size = 1024
	progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
	with open(save_path, 'wb') as file:
		for data in response.iter_content(block_size):
			progress_bar.update(len(data))
			file.write(data)
	progress_bar.close()

def convert_y_unit(y, from_, to_):
	"""
	Arguments:
		y: a list of labels
		from_: 'nM' or 'p'
		to_: 'nM' or 'p'

	Returns:
		y: a numpy array of transformed labels
	"""
	if from_ == 'nM':
		y = y
	elif from_ == 'p':
		y = 10**(-y) / 1e-9

	if to_ == 'p':
		y = -np.log10(y*1e-9 + 1e-10)
	elif to_ == 'nM':
		y = y

	return y

def label_transform(y, binary, threshold, convert_to_log, verbose = True, order = 'descending'):
	"""
	Arguments:
		y: a list of labels
		binary: binarize the label given the threshold
		threshold: threshold values
		convert_to_log: for continuous values such as Kd and etc

	Returns:
		y: a numpy array of transformed labels
	"""

	if (len(np.unique(y)) > 2) and binary:
		if verbose:
			print("Binariztion using threshold' + str(threshold) + ', you use specify your threhsold values by threshold = X)", flush = True, file = sys.stderr)
		if order == 'descending':
			y = np.array([1 if i else 0 for i in np.array(y) < threshold])
		elif order == 'ascending':
			y = np.array([1 if i else 0 for i in np.array(y) > threshold])
		else:
			raise ValueError("Please select order from 'descending or ascending!")
	else:
		if (len(np.unique(y)) > 2) and convert_to_log:
			if verbose:
				print('To log space...', flush = True, file = sys.stderr)
			y = convert_y_unit(np.array(y), 'nM', 'p')
		else:
			y = y

	return y

def convert_to_log(y):
	y = convert_y_unit(np.array(y), 'nM', 'p')
	return y

def convert_back_log(y):
	y = convert_y_unit(np.array(y), 'p', 'nM')
	return y

def binarize(y, threshold, order = 'ascending'):
	if order == 'ascending':
		y = np.array([1 if i else 0 for i in np.array(y) > threshold])
	elif order == 'descending':
		y = np.array([1 if i else 0 for i in np.array(y) < threshold])
	else:
		raise AttributeError("'order' must be either ascending or descending")
	return y

def label_dist(y, name = None):

	try:
		import seaborn as sns
		import matplotlib.pyplot as plt
	except:
		utils.install("seaborn")
		utils.install("matplotlib")
		import seaborn as sns
		import matplotlib.pyplot as plt

	median = np.median(y)
	mean = np.mean(y)

	f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.15, 1)})

	if name is None:
		sns.boxplot(y, ax=ax_box).set_title("Label Distribution")
	else:
		sns.boxplot(y, ax=ax_box).set_title("Label Distribution of " + str(name) + " Dataset")
	ax_box.axvline(median, color='b', linestyle='--')
	ax_box.axvline(mean, color='g', linestyle='--')

	sns.distplot(y, ax = ax_hist)
	ax_hist.axvline(median, color='b', linestyle='--')
	ax_hist.axvline(mean, color='g', linestyle='--')
	ax_hist.legend({'Median':median,'Mean':mean})

	ax_box.set(xlabel='')
	plt.show()
	#print("The median is " + str(median), flush = True, file = sys.stderr)
	#print("The mean is " + str(mean), flush = True, file = sys.stderr)


# random split
def create_fold(df, fold_seed, frac):
	train_frac, val_frac, test_frac = frac
	test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
	train_val = df[~df.index.isin(test.index)]
	val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
	train = train_val[~train_val.index.isin(val.index)]

	return {'train': train.reset_index(drop = True),
			'valid': val.reset_index(drop = True),
			'test': test.reset_index(drop = True)}

# cold setting
def create_fold_setting_cold(df, fold_seed, frac, entity):
	train_frac, val_frac, test_frac = frac
	gene_drop = df[entity].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values

	test = df[df[entity].isin(gene_drop)]

	train_val = df[~df[entity].isin(gene_drop)]

	gene_drop_val = train_val[entity].drop_duplicates().sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed).values
	val = train_val[train_val[entity].isin(gene_drop_val)]
	train = train_val[~train_val[entity].isin(gene_drop_val)]

	return {'train': train.reset_index(drop = True),
			'valid': val.reset_index(drop = True),
			'test': test.reset_index(drop = True)}

# scaffold split
def create_scaffold_split(df, seed, frac, entity):
	# reference: https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py
	try:
		from rdkit import Chem
		from rdkit.Chem.Scaffolds import MurckoScaffold
	except:
		raise ImportError("Please install rdkit by 'conda install -c conda-forge rdkit'! ")
	from tqdm import tqdm
	from random import Random

	from collections import defaultdict
	random = Random(seed)

	s = df[entity].values
	scaffolds = defaultdict(set)
	idx2mol = dict(zip(list(range(len(s))),s))

	error_smiles = 0
	for i, smiles in tqdm(enumerate(s), total=len(s)):
		try:
			scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol = Chem.MolFromSmiles(smiles), includeChirality = False)
			scaffolds[scaffold].add(i)
		except:
			print_sys(smiles + ' returns RDKit error and is thus omitted...')
			error_smiles += 1

	train, val, test = [], [], []
	train_size = int((len(df) - error_smiles) * frac[0])
	val_size = int((len(df) - error_smiles) * frac[1])
	test_size = (len(df) - error_smiles) - train_size - val_size
	train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

	#index_sets = sorted(list(scaffolds.values()), key=lambda i: len(i), reverse=True)
	index_sets = list(scaffolds.values())
	big_index_sets = []
	small_index_sets = []
	for index_set in index_sets:
		if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
			big_index_sets.append(index_set)
		else:
			small_index_sets.append(index_set)
	random.seed(seed)
	random.shuffle(big_index_sets)
	random.shuffle(small_index_sets)
	index_sets = big_index_sets + small_index_sets

	if frac[2] == 0:
		for index_set in index_sets:
			if len(train) + len(index_set) <= train_size:
				train += index_set
				train_scaffold_count += 1
			else:
				val += index_set
				val_scaffold_count += 1
	else:
		for index_set in index_sets:
			if len(train) + len(index_set) <= train_size:
				train += index_set
				train_scaffold_count += 1
			elif len(val) + len(index_set) <= val_size:
				val += index_set
				val_scaffold_count += 1
			else:
				test += index_set
				test_scaffold_count += 1

	return {'train': df.iloc[train].reset_index(drop = True),
			'valid': df.iloc[val].reset_index(drop = True),
			'test': df.iloc[test].reset_index(drop = True)}

def create_combination_split(df, seed, frac):
	"""
	Function for splitting drug combination dataset such that no
	combinations are shared across the split

	:param df: dataset to split as pd Dataframe
	:param seed: random seed
	:param frac: [train, val, test] split fraction as a list
	:return: dictionary of {train, valid, test} datasets
	"""

	# Set split size
	test_size = int(len(df) * frac[2])
	train_size = int(len(df) * frac[0])
	val_size = len(df) - train_size - test_size
	np.random.seed(seed)

	# Create a new column for combination names
	df['concat'] = df['Drug1_ID'] + ',' + df['Drug2_ID']

	# Identify shared drug combinations across all target classes
	combinations = []
	for c in df['Cell_Line_ID'].unique():
		df_cell = df[df['Cell_Line_ID'] == c]
		combinations.append(set(df_cell['concat'].values))

	intxn = combinations[0]
	for c in combinations:
		intxn = intxn.intersection(c)

	# Split combinations into train, val and test
	test_choices = np.random.choice(list(intxn),
						int(test_size / len(df['Cell_Line_ID'].unique())),
						replace=False)
	trainval_intxn = intxn.difference(test_choices)
	val_choices = np.random.choice(list(trainval_intxn),
						int(val_size / len(df['Cell_Line_ID'].unique())),
						replace=False)

	## Create train and test set
	test_set = df[df['concat'].isin(test_choices)].drop(columns=['concat'])
	val_set = df[df['concat'].isin(val_choices)]
	train_set = df[~df['concat'].isin(test_choices)].reset_index(drop=True)
	train_set = train_set[~train_set['concat'].isin(val_choices)]

	return {'train': train_set.reset_index(drop = True),
			'valid': val_set.reset_index(drop = True),
			'test': test_set.reset_index(drop = True)}

def train_val_test_split(len_data, frac, seed):
	test_size = int(len_data * frac[2])
	train_size = int(len_data * frac[0])
	val_size = len_data - train_size - test_size
	np.random.seed(seed)
	x = np.array(list(range(len_data)))
	np.random.shuffle(x)
	return x[:train_size], x[train_size:(train_size + val_size)], x[-test_size:]

def install(package):
	subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def print_sys(s):
	print(s, flush = True, file = sys.stderr)

def _parse_prop(search, proplist):
	"""Extract property value from record using the given urn search filter."""
	props = [i for i in proplist if all(item in i['urn'].items() for item in search.items())]
	if len(props) > 0:
		return props[0]['value'][list(props[0]['value'].keys())[0]]

def request(identifier, namespace='cid', domain='compound', operation=None, output='JSON', searchtype=None):
	"""
	copied from https://github.com/mcs07/PubChemPy/blob/e3c4f4a9b6120433e5cc3383464c7a79e9b2b86e/pubchempy.py#L238
	Construct API request from parameters and return the response.
	Full specification at http://pubchem.ncbi.nlm.nih.gov/pug_rest/PUG_REST.html
	"""
	API_BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
	text_types = str, bytes
	if not identifier:
		raise ValueError('identifier/cid cannot be None')
	# If identifier is a list, join with commas into string
	if isinstance(identifier, int):
		identifier = str(identifier)
	if not isinstance(identifier, text_types):
		identifier = ','.join(str(x) for x in identifier)

	# Build API URL
	urlid, postdata = None, None
	if namespace == 'sourceid':
		identifier = identifier.replace('/', '.')
	if namespace in ['listkey', 'formula', 'sourceid'] \
			or searchtype == 'xref' \
			or (searchtype and namespace == 'cid') or domain == 'sources':
		urlid = quote(identifier.encode('utf8'))
	else:
		postdata = urlencode([(namespace, identifier)]).encode('utf8')
	comps = filter(None, [API_BASE, domain, searchtype, namespace, urlid, operation, output])
	apiurl = '/'.join(comps)
	# Make request
	response = urlopen(apiurl, postdata)
	return response

def NegSample(df, column_names, frac, two_types):
	"""Negative Sampling for Binary Interaction Dataset

	Parameters
	----------
	df : pandas.DataFrame
		Data File
	column_names: list
		column names in the order of [id1, x1, id2, x2]
	"""
	x = int(len(df) * frac)
	id1, x1, id2, x2 = column_names
	df[id1] = df[id1].apply(lambda x: str(x))
	df[id2] = df[id2].apply(lambda x: str(x))

	if not two_types:
		df_unique = np.unique(df[[id1, id2]].values.reshape(-1))
		pos = df[[id1, id2]].values
		pos_set = set([tuple([i[0], i[1]]) for i in pos])
		np.random.seed(1234)
		samples = np.random.choice(df_unique, size=(x, 2), replace=True)
		neg_set = set([tuple([i[0], i[1]]) for i in samples if i[0] != i[1]]) - pos_set

		while len(neg_set) < x:
			sample = np.random.choice(df_unique, 2, replace=False)
			sample = tuple([sample[0], sample[1]])
			if sample not in pos_set:
				neg_set.add(sample)
		neg_list = [list(i) for i in neg_set]

		id2seq = dict(df[[id1, x1]].values)
		id2seq.update(df[[id2, x2]].values)

		neg_list_val = []
		for i in neg_list:
			neg_list_val.append([i[0], id2seq[i[0]], i[1], id2seq[i[1]], 0])

		df = df.append(pd.DataFrame(neg_list_val).rename(columns = {0: id1, 1: x1, 2: id2, 3: x2, 4: 'Y'})).reset_index(drop = True)
		return df
	else:
		df_unique_id1 = np.unique(df[id1].values.reshape(-1))
		df_unique_id2 = np.unique(df[id2].values.reshape(-1))

		pos = df[[id1, id2]].values
		pos_set = set([tuple([i[0], i[1]]) for i in pos])
		np.random.seed(1234)

		sample_id1 = np.random.choice(df_unique_id1, size=len(df), replace=True)
		sample_id2 = np.random.choice(df_unique_id2, size=len(df), replace=True)

		neg_set = set([tuple([sample_id1[i], sample_id2[i]]) for i in range(len(df)) if sample_id1[i] != sample_id2[i]]) - pos_set

		while len(neg_set) < len(df):
			sample_id1 = np.random.choice(df_unique_id1, size=1, replace=True)
			sample_id2 = np.random.choice(df_unique_id2, size=1, replace=True)

			sample = tuple([sample_id1[0], sample_id2[0]])
			if sample not in pos_set:
				neg_set.add(sample)
		neg_list = [list(i) for i in neg_set]

		id2seq1 = dict(df_temp[[id1, x1]].values)
		id2seq2 = dict(df_temp[[id2, x2]].values)

		neg_list_val = []
		for i in neg_list:
			neg_list_val.append([i[0], id2seq[i[0]], i[1], id2seq[i[1]], 0])

		df = df.append(pd.DataFrame(neg_list_val).rename(columns = {0: id1, 1: x1, 2: id2, 3: x2, 4: 'Y'})).reset_index(drop = True)
		return df

def uniprot2seq(ProteinID):
	"""Get protein sequence from Uniprot ID

	Parameters
	----------
	ProteinID : str
		Uniprot ID

	Returns
	-------
	str
		Amino acid sequence of input uniprot ID
	"""
	import urllib
	import string
	import urllib.request as ur

	ID = str(ProteinID)
	localfile = ur.urlopen('http://www.uniprot.org/uniprot/' + ID + '.fasta')
	temp = localfile.readlines()
	res = ''
	for i in range(1, len(temp)):
		res = res + temp[i].strip().decode("utf-8")
	return res

def cid2smiles(cid):
	try:
		smiles = _parse_prop({'label': 'SMILES', 'name': 'Canonical'}, json.loads(request(cid).read().decode())['PC_Compounds'][0]['props'])
	except:
		print('cid ' + str(cid) + ' failed, use NULL string')
		smiles = 'NULL'
	return smiles

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

def to_submission_format(results):
	df = pd.DataFrame(results)
	def get_metric(x):
		metric = []
		for i in x:
			metric.append(list(i.values())[0])
		return [round(np.mean(metric), 3), round(np.std(metric), 3)]
	return dict(df.apply(get_metric, axis = 1))
