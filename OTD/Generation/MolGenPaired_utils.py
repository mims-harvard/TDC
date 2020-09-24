import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import wget 

def read_paired_smiles_file(filename):
	with open(filename, 'r') as fin:
		lines = fin.readlines()
	x_lst = [line.split()[0] for line in lines]
	y_lst = [line.strip().split()[1] for line in lines]
	return x_lst, y_lst


def qed_process(name, path):
	if not os.path.exists(path):
		os.makedirs(path)
	data_file = os.path.join(path, name+'.txt')
	if not os.path.exists(data_file):
		download_file = "https://raw.githubusercontent.com/wengong-jin/iclr19-graph2graph/master/data/qed/train_pairs.txt"
		wget.download(download_file, data_file)
	x_lst, y_lst = read_paired_smiles_file(data_file)
	return x_lst, y_lst 

def drd2_process(name, path):
	if not os.path.exists(path):
		os.makedirs(path)
	data_file = os.path.join(path, name+'.txt')
	if not os.path.exists(data_file):
		download_file = "https://raw.githubusercontent.com/wengong-jin/iclr19-graph2graph/master/data/drd2/train_pairs.txt"
		wget.download(download_file, data_file)
	x_lst, y_lst = read_paired_smiles_file(data_file)
	return x_lst, y_lst 	


def logp_process(name, path):
	if not os.path.exists(path):
		os.makedirs(path)
	data_file = os.path.join(path, name+'.txt')
	if not os.path.exists(data_file):
		download_file = "https://raw.githubusercontent.com/wengong-jin/iclr19-graph2graph/master/data/logp04/train_pairs.txt"
		wget.download(download_file, data_file)
	x_lst, y_lst = read_paired_smiles_file(data_file)
	return x_lst, y_lst 	







