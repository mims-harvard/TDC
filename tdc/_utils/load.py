import requests
from zipfile import ZipFile 
import os, sys
import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm

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

def zip_data_download_wrapper(name, path, dataset_names):
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

def oracle_download_wrapper(name, path, oracle_names):
	name = fuzzy_search(name, oracle_names)
	if name in trivial_oracle_names:
		return name

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
		elif name2type[name] == 'zip':
			df = pd.read_pickle(os.path.join(path, name + '/' + name + '.pkl'))
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

def interaction_dataset_load(name, path, target, dataset_names, aux_column):
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

def three_dim_dataset_load(name, path, dataset_names):
	name = zip_data_download_wrapper(name, path, dataset_names)
	print_sys('Loading...')
	df = pd_load(name, path)
	return df, os.path.join(path, name), name

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