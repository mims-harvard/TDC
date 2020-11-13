import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

# from. evaluator import Evaluator
from .utils import * 
from .metadata import download_oracle_names, oracle_names, molecule_evaluator_name
from .chem_utils import novelty, diversity, unique_rate, validity_ratio
from .chem_utils import penalized_logp, qed, drd2, SA, gsk3b, jnk3, askcos, ibm_rxn
from .chem_utils import celecoxib_rediscovery, troglitazone_rediscovery, thiothixene_rediscovery
from .chem_utils import aripiprazole_similarity, albuterol_similarity, mestranol_similarity, median1, median2


class Oracle:
	def __init__(self, name):
		name = fuzzy_search(name, oracle_names)
		if name in download_oracle_names:
			self.name = oracle_load(name)
		else:
			self.name = name
		self.evaluator_func = None
		self.assign_evaluator() 

	def assign_evaluator(self):
		'''
			self.name -> self.evaluator_func
			assert self.name in ['logp', 'drd', ...]
		'''
		if self.name == 'novelty':
			self.evaluator_func = novelty  
		elif self.name == 'diversity':
			self.evaluator_func = diversity 
		elif self.name == 'validity':
			self.evaluator_func = validity_ratio 
		elif self.name == 'uniqueness':
			self.evaluator_func = unique_rate 
		elif self.name == 'logp':
			self.evaluator_func = penalized_logp 
		elif self.name == 'qed':
			self.evaluator_func = qed  
		elif self.name == 'drd2':
			self.evaluator_func = drd2 
		elif self.name == 'sa':
			self.evaluator_func = SA 
		elif self.name == 'gsk3b':
			oracle_object = gsk3b
			self.evaluator_func = oracle_object
		elif self.name == 'jnk3':
			oracle_object = jnk3()
			self.evaluator_func = oracle_object
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
		elif self.name == 'askcos':
			self.evaluator_func = askcos
		elif self.name == 'ibm_rxn':
			self.evaluator_func = ibm_rxn
		else:
			return 

	def __call__(self, smiles, temp = None):
		if temp is None:
			if type(smiles)==list:
				if self.name in molecule_evaluator_name: 
					#### evaluator for distribution learning, e.g., diversity, validity
					#### the input of __call__ is list of smiles
					return self.evaluator_func(smiles) 
				else:
					#### evaluator for single molecule, 
					#### the input of __call__ is a single smiles OR list of smiles
					return list(map(self.evaluator_func, smiles))
			else: ### type(smiles)==str:
				return self.evaluator_func(smiles)
		else:
			# novelty
			return self.evaluator_func(smiles, temp)

'''
guacamol_oracle = ['celecoxib_rediscovery', 'troglitazone_rediscovery', 'thiothixene_rediscovery', \
				   'aripiprazole_similarity', 'albuterol_similarity', 'mestranol_similarity', 
				   'C11H24_isomer', 'C9H10N2O2PF2Cl_isomor', \
				   'median_molecule_1', 'median_molecule_2', \
				   'Osimertinib_MPO', 'Fexofenadine_MPO', 'Ranolazine_MPO', 'Perindopril_MPO', \
				   'Amlodipine_MPO', 'Sitagliptin_MPO', 'Zaleplon_MPO', \
				   'Valsartan_SMARTS', 'deco_hop', 'scaffold_hop']


'''


