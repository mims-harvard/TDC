import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .. import base_dataset
from ..utils import *

class DataLoader(base_dataset.DataLoader):
	def __init__(self, name, path, target, print_stats, dataset_names):
		entity1, entity2, raw_y, entity1_idx, entity2_idx = interaction_dataset_load(name, path, target)
		
		self.name = name
		self.entity1 = entity1
		self.entity2 = entity2
		self.raw_y = raw_y
		self.y = raw_y
		self.entity1_idx = entity1_idx
		self.entity2_idx = entity2_idx
		self.path = path
		self.file_format = 'csv'
		self.target = target
		
		self.entity1_name = 'Entity1'
		self.entity2_name = 'Entity2'

	def get_data(self, format = 'df'):
		'''
		Arguments:
			df: return pandas DataFrame; if not true, return np.arrays			
		returns:
			self.drugs: drug smiles strings np.array
			self.targets: target Amino Acid Sequence np.array
			self.y: inter   action score np.array
		'''
		if format == 'df':
			return pd.DataFrame({self.entity1_name + '_ID': self.entity1_idx, self.entity1_name: self.entity1, self.entity2_name + '_ID': self.entity2_idx, self.entity2_name: self.entity2, 'Y': self.y})
		elif format == 'DeepPurpose':
			return self.entity1, self.entity2, self.y
		elif format == 'dict':			
			return {self.entity1_name + '_ID': self.entity1_idx, self.entity1_name: self.entity1, self.entity2_name + '_ID': self.entity2_idx, self.entity2_name: self.entity2, 'Y': self.y}
		else:
			raise AttributeError("Please use the correct format input")

	def print_stats(self):
		print('There are ' + str(len(np.unique(self.entity1))) + ' unique ' + self.entity1_name.lower() + 's.', flush = True, file = sys.stderr)
		print('There are ' + str(len(np.unique(self.entity2))) + ' unique ' + self.entity2_name.lower() + 's.', flush = True, file = sys.stderr)
		print('There are ' + str(len(self.y)) + ' ' + self.entity1_name.lower() + '-' + self.entity2_name.lower() + ' pairs.', flush = True, file = sys.stderr)

	def get_split(self, method = 'random', seed = 'benchmark', frac = [0.7, 0.1, 0.2]):
		'''
		Arguments:
			method: splitting schemes: random, cold_drug, cold_target
			seed: 'benchmark' seed set to 1234, or int values
			frac: train/val/test split
		'''
		if seed == 'benchmark':
			seed = 1234

		df = self.get_data(df = True)

		if method == 'random':
			return create_fold(df, seed, frac)
		elif method == 'cold_' + self.entity1_name.lower():
			return create_fold_setting_cold(df, seed, frac, self.entity1_name)
		elif method == 'cold_' + self.entity2_name.lower():
			return create_fold_setting_cold(df, seed, frac, self.entity2_name)

	def to_graph(self, threshold = None, format = 'edge_list', split = True, frac = [0.7, 0.1, 0.2], seed = 'benchmark'):
		'''
		Arguments:
			format: edge_list / dgl / pyg df object			
		'''
		if seed == 'benchmark':
			seed = 1234

		df = self.get_data(df = True)

		if threshold is None:
			raise AttributeError("Please specify the threshold to binarize the data by 'to_graph(threshold = N)'!")

		if len(np.unique(self.raw_y)) > 2:
			print("The dataset label consists of affinity scores. Binarization using threshold " + str(threshold) + " is conducted to construct the positive edges in the network. Adjust the threshold by to_graph(threshold = X)", flush = True, file = sys.stderr)

		df['label_binary'] = label_transform(self.raw_y, True, threshold, False, verbose  = False)
		df_pos = df[df.label_binary == 1]
		df_neg = df[df.label_binary == 0]

		return_dict = {}

		pos_edges = df_pos[[self.entity1_name + '_ID', self.entity2_name + '_ID']].values
		neg_edges = df_neg[[self.entity1_name + '_ID', self.entity2_name + '_ID']].values
		edges = df[[self.entity1_name + '_ID', self.entity2_name + '_ID']].values

		if format == 'edge_list':
			return_dict['edge_list'] = pos_edges
			return_dict['neg_edges'] = neg_edges
		elif format == 'dgl':
			try:
				import dgl
			except:
				install("dgl")
				import dgl
			unique_entities = np.unique(pos_edges.T.flatten()).tolist()
			index = list(range(len(unique_entities)))
			dict_ = dict(zip(unique_entities, index))
			edge_list1 = np.array([dict_[i] for i in pos_edges.T[0]])
			edge_list2 = np.array([dict_[i] for i in pos_edges.T[1]])
			return_dict['dgl_graph'] = dgl.DGLGraph((edge_list1, edge_list2))
			return_dict['index_to_entities'] = dict_
			
		elif format == 'pyg':
			try:
				import torch
				from torch_geometric.data import Data
			except:
				raise ImportError("Please see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html to install pytorch geometric!")
			
			unique_entities = np.unique(pos_edges.T.flatten()).tolist()
			index = list(range(len(unique_entities)))
			dict_ = dict(zip(unique_entities, index))
			edge_list1 = np.array([dict_[i] for i in pos_edges.T[0]])
			edge_list2 = np.array([dict_[i] for i in pos_edges.T[1]])

			edge_index = torch.tensor([edge_list1, edge_list2], dtype=torch.long)
			x = torch.tensor(np.array(index), dtype=torch.float)
			data = Data(x=x, edge_index=edge_index)
			return_dict['pyg_graph'] = data
			return_dict['index_to_entities'] = dict_
			
		elif format == 'df':
			return_dict['df'] = df
		
		if split:
			return_dict['split'] = create_fold(df, seed, frac)

		return return_dict


