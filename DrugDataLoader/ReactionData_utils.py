import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import wget 


def file2reaction(filename):
	with open(filename, 'r') as fin:
		lines = fin.readlines()[1:]
	reactions_lst = [line.split()[0] for line in lines]
	reactant_lst = [i.split('>')[0] for i in reactions_lst]
	catalyst_lst = [i.split('>')[1] for i in reactions_lst]
	product_lst = [i.split('>')[2] for i in reactions_lst]
	return reactant_lst, catalyst_lst, product_lst  

def uspto_process(name, path):
	if not os.path.exists(path):
		os.makedirs(path)
	data_file = os.path.join(path, name+'.txt')
	reactant_lst, catalyst_lst, product_lst = file2reaction(data_file)
	return reactant_lst, catalyst_lst, product_lst  



