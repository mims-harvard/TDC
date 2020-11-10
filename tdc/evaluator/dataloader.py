import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from ..utils import * 
from ..chem_utils import penalized_logp, qed, drd2, SA, gsk3, jnk3  
from ..chem_utils import novelty, diversity, unique_rate, validity_ratio
from ..chem_utils import celecoxib_rediscovery, troglitazone_rediscovery, thiothixene_rediscovery
from ..chem_utils import aripiprazole_similarity, albuterol_similarity, mestranol_similarity, median1, median2
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
		

		elif self.name == 'roc-auc':
			pass 
		elif self.name == 'f1':
			pass 
		elif self.name == 'pr-auc':
			pass 
		elif self.name == 'precision':
			pass 
		elif self.name == 'recall':
			pass 
		elif self.name == 'accuracy':
			pass


	def __call__(self, first_argu, groundtruth = None):
		if groundtruth is None:
			smiles = first_argu
			if type(smiles)==list:
				return list(map(self.evaluator_func, smiles))
			else: ### type(smiles)==str:
				return self.evaluator_func(smiles)
		else:
			prediction = first_argu
			return evaluator_func(prediction, groundtruth)


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
		elif self.name == 'celecoxib_rediscovery':
			self.evaluator_func = celecoxib_rediscovery
		elif self.name == 'troglitazone_rediscovery':
			self.evaluator_func = troglitazone_rediscovery
		elif self.name == 'thiothixene_rediscovery':
			self.evaluator_func = thiothixene_rediscovery
		elif self.name == 'aripiprazole_similarity':
			self.evaluator_func = aripiprazole_similarity
		elif self.name == 'albuterol_similarity':
			self.evaluator_func = albuterol_similarity
		elif self.name == 'mestranol_similarity':
			self.evaluator_func = mestranol_similarity
		elif self.name == 'median1':
			self.evaluator_func = median1
		elif self.name == 'median2':
			self.evaluator_func = median2
		else:
			return 


'''
guacamol_oracle = ['celecoxib_rediscovery', 'troglitazone_rediscovery', 'thiothixene_rediscovery', \
				   'aripiprazole_similarity', 'albuterol_similarity', 'mestranol_similarity', 
				   'C11H24_isomer', 'C9H10N2O2PF2Cl_isomor', \
				   'median_molecule_1', 'median_molecule_2', \
				   'Osimertinib_MPO', 'Fexofenadine_MPO', 'Ranolazine_MPO', 'Perindopril_MPO', \
				   'Amlodipine_MPO', 'Sitagliptin_MPO', 'Zaleplon_MPO', \
				   'Valsartan_SMARTS', 'deco_hop', 'scaffold_hop']


'''


