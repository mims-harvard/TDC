"""wrapper for download various dataset 
"""
import requests
from zipfile import ZipFile 
import os, sys
import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm

from ..metadata import name2type, name2id, dataset_list, dataset_names, benchmark_names, benchmark2id, benchmark2type
from ..metadata import property_names, paired_dataset_names, single_molecule_dataset_names
from ..metadata import retrosyn_dataset_names, forwardsyn_dataset_names, molgenpaired_dataset_names, generation_datasets
from ..metadata import oracle2id, receptor2id, download_oracle_names, trivial_oracle_names, oracle_names, oracle2type 
from collections import defaultdict 

receptor_names = list(receptor2id.keys())
sys.path.append('../')

from .misc import fuzzy_search, print_sys

def download_wrapper(name, path, dataset_names):
	"""wrapper for downloading a dataset given the name and path, for csv,pkl,tsv files
	
	Args:
	    name (str): the rough dataset query name
	    path (str): the path to save the dataset
	    dataset_names (list): the list of available dataset names to search the query dataset
	
	Returns:
	    str: the exact dataset query name
	"""
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

def zip_data_download_wrapper(name, path, dataset_names):
	"""wrapper for downloading a dataset given the name and path - zip file, automatically unzipping
	
	Args:
	    name (str): the rough dataset query name
	    path (str): the path to save the dataset
	    dataset_names (list): the list of available dataset names to search the query dataset
	
	Returns:
	    str: the exact dataset query name
	"""
	name = fuzzy_search(name, dataset_names)
	server_path = 'https://dataverse.harvard.edu/api/access/datafile/'

	dataset_path = server_path + str(name2id[name])

	if not os.path.exists(path):
		os.mkdir(path)

	if os.path.exists(os.path.join(path, name)):
		print_sys('Found local copy...')
	else:
		print_sys('Downloading...')
		dataverse_download(dataset_path, path, name, name2type)
		print_sys('Extracting zip file...')
		with ZipFile(os.path.join(path, name + '.zip'), 'r') as zip:
			zip.extractall(path = os.path.join(path))
		print_sys("Done!")
	return name

def dataverse_download(url, path, name, types):
	"""dataverse download helper with progress bar
	
	Args:
	    url (str): the url of the dataset
	    path (str): the path to save the dataset
	    name (str): the dataset name
	    types (dict): a dictionary mapping from the dataset name to the file format
	"""
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


def oracle_download_wrapper(name, path, oracle_names):
	"""wrapper for downloading an oracle model checkpoint given the name and path
	
	Args:
	    name (str): the rough oracle query name
	    path (str): the path to save the oracle
	    dataset_names (list): the list of available exact oracle names
	
	Returns:
	    str: the exact oracle query name
	"""
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


def receptor_download_wrapper(name, path):
	"""wrapper for downloading an receptor pdb file given the name and path
	
	Args:
	    name (str): the exact pdbid
	    path (str): the path to save the oracle
	
	Returns:
	    str: the exact pdbid
	"""

	server_path = 'https://dataverse.harvard.edu/api/access/datafile/'
	dataset_paths = [server_path + str(receptor2id[name][0]), server_path + str(receptor2id[name][1])] 

	if not os.path.exists(path):
		os.mkdir(path)

	if os.path.exists(os.path.join(path, name + '.pdbqt')) and os.path.exists(os.path.join(path, name + '.pdb')):
		print_sys('Found local copy...')
	else:
		print_sys("Downloading receptor...")
		receptor2type = defaultdict(lambda:'pdbqt')
		dataverse_download(dataset_paths[0], path, name, receptor2type) ## to-do to-check
		receptor2type = defaultdict(lambda:'pdb')
		dataverse_download(dataset_paths[1], path, name, receptor2type) ## to-do to-check
		print_sys("Done!")
	return name




def bm_download_wrapper(name, path):
	"""wrapper for downloading a benchmark group given the name and path
	
	Args:
	    name (str): the rough benckmark group query name
	    path (str): the path to save the benchmark group
	    dataset_names (list): the list of available benchmark group names
	
	Returns:
	    str: the exact benchmark group query name
	"""
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
	"""load a pandas dataframe from local file.
	
	Args:
	    name (str): dataset name
	    path (str): the path where the dataset is saved
	
	Returns:
	    pandas.DataFrame: loaded dataset in dataframe
	
	Raises:
	    ValueError: the file format is not supported. currently only support tab/csv/pkl/zip
	"""
	try:
		if name2type[name] == 'tab':
			df = pd.read_csv(os.path.join(path, name + '.' + name2type[name]), sep = '\t')
		elif name2type[name] == 'csv':
			df = pd.read_csv(os.path.join(path, name + '.' + name2type[name]))
		elif name2type[name] == 'pkl':
			df = pd.read_pickle(os.path.join(path, name + '.' + name2type[name]))
		elif name2type[name] == 'zip':
			df = pd.read_pickle(os.path.join(path, name + '/' + name + '.pkl'))
		else:
			raise ValueError("The file type must be one of tab/csv/pickle/zip.")
		try:
			df = df.drop_duplicates()
		except:
			pass
		return df
	except (EmptyDataError, EOFError) as e:
		import sys
		sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")

def property_dataset_load(name, path, target, dataset_names):
	"""a wrapper to download, process and load single-instance prediction task datasets
	
	Args:
	    name (str): the rough dataset name
	    path (str): the dataset path to save/retrieve
	    target (str): for multi-label dataset, retrieve the label of interest
	    dataset_names (list): a list of availabel exact dataset names
	
	Returns:
	    pandas.Series: three series (entity representation, label, entity id)
	"""
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
				sys.exit("Please report this error to contact@tdcommons.ai, thanks!")
	try:
		return df['X'], df[target], df['ID']
	except:
		return df['Drug'], df[target], df['Drug_ID']

def interaction_dataset_load(name, path, target, dataset_names, aux_column):
	"""a wrapper to download, process and load two-instance prediction task datasets
	
	Args:
	    name (str): the rough dataset name
	    path (str): the dataset path to save/retrieve
	    target (str): for multi-label dataset, retrieve the label of interest
	    dataset_names (list): a list of availabel exact dataset names
	
	Returns:
	    pandas.Series: three series (entity 1 representation, entity 2 representation, entity id 1, entity id 2, label)
	"""
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
		if aux_column is None:
			return df['X1'], df['X2'], df[target], df['ID1'], df['ID2'], '_'
		else:
			return df['X1'], df['X2'], df[target], df['ID1'], df['ID2'], df[aux_column]

	except:
		with open(os.path.join(path, name + '.' + name2type[name]), 'r') as f:
			flag = 'Service Unavailable' in ' '.join(f.readlines())
			if flag:
				import sys
				sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")
			else:
				import sys
				sys.exit("Please report this error to cosamhkx@gmail.com, thanks!")


def multi_dataset_load(name, path, dataset_names):
	"""a wrapper to download, process and load multiple(>2)-instance prediction task datasets. assume the downloaded file is already processed
	
	Args:
	    name (str): the rough dataset name
	    path (str): the dataset path to save/retrieve
	    target (str): for multi-label dataset, retrieve the label of interest
	    dataset_names (list): a list of availabel exact dataset names
	
	Returns:
	    pandas.DataFrame: the raw dataframe
	"""
	name = download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	return df

def generation_paired_dataset_load(name, path, dataset_names, input_name, output_name):
	"""a wrapper to download, process and load generation-paired task datasets
	
	Args:
	    name (str): the rough dataset name
	    path (str): the dataset path to save/retrieve
	    target (str): for multi-label dataset, retrieve the label of interest
	    dataset_names (list): a list of availabel exact dataset names
	
	Returns:
	    pandas.Series: two series (entity 1 representation, label)
	"""
	name = download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	return df[input_name], df[output_name]

def three_dim_dataset_load(name, path, dataset_names):
	"""a wrapper to download, process and load 3d molecule task datasets
	
	Args:
	    name (str): the rough dataset name
	    path (str): the dataset path to save/retrieve
	    dataset_names (list): a list of availabel exact dataset names
	
	Returns:
	    pandas.DataFrame: the dataframe holds 3d information
	    str: the path of the dataset
	    str: the name of the dataset
	"""
	name = zip_data_download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	return df, os.path.join(path, name), name

def distribution_dataset_load(name, path, dataset_names, column_name):
	"""a wrapper to download, process and load molecule distribution learning task datasets. assume the downloaded file is already processed
	
	Args:
	    name (str): the rough dataset name
	    path (str): the dataset path to save/retrieve
	    dataset_names (list): a list of availabel exact dataset names
	    column_name (str): the column specifying where molecule locates
	
	Returns:
	    pandas.Series: the input list of molecules representation
	"""
	name = download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	return df[column_name]

def generation_dataset_load(name, path, dataset_names):
	"""a wrapper to download, process and load generation task datasets. assume the downloaded file is already processed
	
	Args:
	    name (str): the rough dataset name
	    path (str): the dataset path to save/retrieve
	    dataset_names (list): a list of availabel exact dataset names
	
	Returns:
	    pandas.Series: the data series
	"""
	name = download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	return df['input'], df['target']

def oracle_load(name, path = './oracle', oracle_names = oracle_names):
	"""a wrapper to download, process and load oracles. 
	
	Args:
	    name (str): the rough oracle name
	    path (str): the oracle path to save/retrieve, defaults to './oracle'
	    dataset_names (list): a list of availabel exact oracle names
	
	Returns:
	    str: exact oracle name
	"""
	name = oracle_download_wrapper(name, path, oracle_names)
	return name


def receptor_load(name, path = './oracle'):
	"""a wrapper to download, process and load pdb file. 
	
	Args:
	    name (str): the rough pdbid name
	    path (str): the oracle path to save/retrieve, defaults to './oracle'
	
	Returns:
	    str: exact pdbid name
	"""
	name = receptor_download_wrapper(name, path)
	return name	


def bm_group_load(name, path):
	"""a wrapper to download, process and load benchmark group
	
	Args:
	    name (str): the rough benchmark group name
	    path (str): the benchmark group path to save/retrieve
	
	Returns:
	    str: exact benchmark group name
	"""
	name = bm_download_wrapper(name, path)
	return name