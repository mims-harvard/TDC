import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from ..utils import * 
from ..chem_utils import penalized_logp, qed, drd2, SA, gsk3, jnk3  
from ..chem_utils import novelty, diversity, unique_rate, validity_ratio
from ..metadata import evaluator_name



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

	def __call__(self, smiles):
		if type(smiles)==list:
			return list(map(self.evaluator_func, smiles))
		else: ### type(smiles)==str:
			return self.evaluator_func(smiles)



class Oracle(Evaluator):
	def __init__(self, name):
		from ..utils import oracle_names ### e.g., ['drd2', 'qed', 'logp']
		self.oracle_name = oracle_names
		self.name = oracle_load(name)
		#### plogp, drd, .... 
		self.evaluator_func = lambda x:1 
		self.assign_evaluator() 


	def assign_evaluator(self):
		'''
			self.name -> self.evaluator_func
			assert self.name in ['logp', 'drd', ...]
		'''
		if self.name == 'logp':
			self.evaluator_func = penalized_logp 
		elif self.name == 'qed':
			self.evaluator_func = qed  
		elif self.name == 'drd2':
			self.evaluator_func = drd2 
		elif self.name == 'sa':
			self.evaluator_func = SA 
		elif self.name == 'gsk3':
			self.evaluator_func = gsk3 
		elif self.name == 'jnk3':
			self.evaluator_func = jnk3 
		else:
			return 



