import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .utils import * 
from .chem_utils import novelty, diversity, unique_rate, validity_ratio
from .metadata import evaluator_name
try:
	from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score
except:
	ImportError("Please install sklearn by 'conda install -c anaconda scikit-learn' or 'pip install scikit-learn '! ")

class Evaluator:
	def __init__(self, name, molecule_base = None):
		self.name = fuzzy_search(name, evaluator_name)
		self.molecule_base = molecule_base
		self.assign_evaluator()

	def assign_evaluator(self):
		if self.name == 'novelty':
			self.evaluator_func = novelty  
		elif self.name == 'diversity':
			self.evaluator_func = diversity 
		elif self.name == 'validity':
			self.evaluator_func = validity_ratio 
		elif self.name == 'unique':
			self.evaluator_func = unique_rate 
		

		elif self.name == 'roc-auc':
			self.evaluator_func = roc_auc_score 
		elif self.name == 'f1':
			self.evaluator_func = f1_score 
		elif self.name == 'pr-auc':
			self.evaluator_func = average_precision_score 
		elif self.name == 'precision':
			self.evaluator_func = precision_score
		elif self.name == 'recall':
			self.evaluator_func = recall_score
		elif self.name == 'accuracy':
			self.evaluator_func = accuracy_score 


	def __call__(self, first_argu, groundtruth = None):
		if groundtruth is None:
			smiles = first_argu
			if type(smiles)==list:
				return list(map(self.evaluator_func, smiles))
			else: ### type(smiles)==str:
				return self.evaluator_func(smiles)
		else:
			prediction = first_argu
			if self.name in ['f1', 'pr-auc', 'precision', 'recall', 'accuracy']:
				binarize = lambda x:1 if x>0.5 else 0
				prediction = list(map(binarize, prediction))
			return self.evaluator_func(groundtruth, prediction)

			'''
				f1_score(label_all, predict_all)  ## binary
				average_precision_score(label_all, predict_all)  ## binary 

			'''