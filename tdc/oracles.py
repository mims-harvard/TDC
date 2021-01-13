import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .utils import * 
from .metadata import download_oracle_names, oracle_names, distribution_oracles

class Oracle:
	def __init__(self, name, target_smiles = None, **kwargs):
		self.target_smiles = target_smiles
		self.kwargs = kwargs

		name = fuzzy_search(name, oracle_names)
		if name in download_oracle_names:
			self.name = oracle_load(name)
		else:
			self.name = name
		self.evaluator_func = None
		self.assign_evaluator() 

	def assign_evaluator(self):		
		if self.name == 'logp':			############################ molecular property 
			from .chem_utils import penalized_logp
			self.evaluator_func = penalized_logp 
		elif self.name == 'qed':
			from .chem_utils import qed
			self.evaluator_func = qed  
		elif self.name == 'drd2':
			from .chem_utils import drd2
			self.evaluator_func = drd2 
		elif self.name == 'sa':
			from .chem_utils import SA
			self.evaluator_func = SA 
		elif self.name == 'gsk3b':
			from .chem_utils import gsk3b
			oracle_object = gsk3b
			self.evaluator_func = oracle_object
		elif self.name == 'jnk3':
			from .chem_utils import jnk3
			oracle_object = jnk3()
			self.evaluator_func = oracle_object
		elif self.name == 'similarity_meta':	############################ oracle meta
			from .chem_utils import similarity_meta
			self.evaluator_func = similarity_meta(target_smiles = self.target_smiles, **self.kwargs)
		elif self.name == 'rediscovery_meta':
			from .chem_utils import rediscovery_meta 
			self.evaluator_func = rediscovery_meta(target_smiles = self.target_smiles, **self.kwargs)
		elif self.name == 'isomer_meta':
			from .chem_utils import isomer_meta 
			self.evaluator_func = isomer_meta(target_smiles = self.target_smiles, **self.kwargs)
		elif self.name == 'median_meta':
			from .chem_utils import median_meta 
			self.evaluator_func = median_meta(target_smiles_1 = self.target_smiles[0], 
											  target_smiles_2 = self.target_smiles[1], 
											  **self.kwargs) 
		elif self.name == 'rediscovery':	############################ guacamol 
			from .chem_utils import celecoxib_rediscovery, troglitazone_rediscovery, thiothixene_rediscovery
			self.evaluator_func = {"Celecoxib": celecoxib_rediscovery, 
								"Troglitazone": troglitazone_rediscovery, 
								"Thiothixene": thiothixene_rediscovery}
		elif self.name == 'celecoxib_rediscovery':
			from .chem_utils import celecoxib_rediscovery
			self.evaluator_func = celecoxib_rediscovery
		elif self.name == 'troglitazone_rediscovery':
			from .chem_utils import troglitazone_rediscovery
			self.evaluator_func = troglitazone_rediscovery
		elif self.name == 'thiothixene_rediscovery':
			from .chem_utils import thiothixene_rediscovery
			self.evaluator_func = thiothixene_rediscovery
		elif self.name == 'similarity':
			from .chem_utils import aripiprazole_similarity, albuterol_similarity, mestranol_similarity
			self.evaluator_func = {"Aripiprazole": aripiprazole_similarity,
									"Albuterol": albuterol_similarity,
									"Mestranol": mestranol_similarity}
		elif self.name == 'aripiprazole_similarity':
			from .chem_utils import aripiprazole_similarity
			self.evaluator_func = aripiprazole_similarity
		elif self.name == 'albuterol_similarity':
			from .chem_utils import albuterol_similarity
			self.evaluator_func = albuterol_similarity
		elif self.name == 'mestranol_similarity':
			from .chem_utils import mestranol_similarity
			self.evaluator_func = mestranol_similarity
		elif self.name == 'median':
			from .chem_utils import median1, median2
			self.evaluator_func = {'Median 1': median1,
									'Median 2': median2}
		elif self.name == 'median1':
			from .chem_utils import median1
			self.evaluator_func = median1
		elif self.name == 'median2':
			from .chem_utils import median2
			self.evaluator_func = median2
		elif self.name == 'mpo':
			from .chem_utils import osimertinib_mpo, fexofenadine_mpo, ranolazine_mpo, perindopril_mpo, amlodipine_mpo, sitagliptin_mpo, zaleplon_mpo
			self.evaluator_func = {'Osimertinib': osimertinib_mpo,
									'Fexofenadine': fexofenadine_mpo,
									'Ranolazine': ranolazine_mpo,
									'Perindopril': perindopril_mpo,
									'Amlodipine': amlodipine_mpo,
									'Sitagliptin': sitagliptin_mpo,
									'Zaleplon': zaleplon_mpo}
		elif self.name == 'osimertinib_mpo':
			from .chem_utils import osimertinib_mpo
			self.evaluator_func = osimertinib_mpo
		elif self.name == 'fexofenadine_mpo':
			from .chem_utils import fexofenadine_mpo
			self.evaluator_func = fexofenadine_mpo
		elif self.name == 'ranolazine_mpo':
			from .chem_utils import ranolazine_mpo
			self.evaluator_func = ranolazine_mpo
		elif self.name == 'perindopril_mpo':
			from .chem_utils import perindopril_mpo
			self.evaluator_func = perindopril_mpo
		elif self.name == 'amlodipine_mpo':
			from .chem_utils import amlodipine_mpo
			self.evaluator_func = amlodipine_mpo
		elif self.name == 'sitagliptin_mpo':
			from .chem_utils import sitagliptin_mpo
			self.evaluator_func = sitagliptin_mpo
		elif self.name == 'zaleplon_mpo':
			from .chem_utils import zaleplon_mpo
			self.evaluator_func = zaleplon_mpo
		elif self.name == 'valsartan_smarts':
			from .chem_utils import valsartan_smarts
			self.evaluator_func = valsartan_smarts
		elif self.name == 'hop':
			from .chem_utils import deco_hop, scaffold_hop
			self.evaluator_func = {'Deco Hop': deco_hop,
									'Scaffold Hop': scaffold_hop}
		elif self.name == 'deco_hop':
			from .chem_utils import deco_hop
			self.evaluator_func = deco_hop
		elif self.name == 'scaffold_hop':
			from .chem_utils import scaffold_hop
			self.evaluator_func = scaffold_hop
		elif self.name == 'isomers_c7h8n2o2':
			from .chem_utils import isomers_c7h8n2o2
			self.evaluator_func = isomers_c7h8n2o2
		elif self.name == 'isomers_c9h10n2o2pf2cl':
			from .chem_utils import isomers_c9h10n2o2pf2cl
			self.evaluator_func = isomers_c9h10n2o2pf2cl  
		elif self.name == 'isomers':
			from .chem_utils import isomers_c7h8n2o2, isomers_c9h10n2o2pf2cl
			self.evaluator_func = {'c7h8n2o2': isomers_c7h8n2o2,
									'c9h10n2o2pf2cl': isomers_c9h10n2o2pf2cl}
		elif self.name == 'askcos':  		#### synthetic analysis 
			from .chem_utils import askcos
			self.evaluator_func = askcos
		elif self.name == 'ibm_rxn':
			from .chem_utils import ibm_rxn
			self.evaluator_func = ibm_rxn
		elif self.name == 'molecule_one_synthesis':
			from .chem_utils import molecule_one_retro
			self.evaluator_func = molecule_one_retro(**self.kwargs)
		elif self.name == 'docking_score':
			from .chem_utils import docking_meta
			self.evaluator_func = docking_meta(**self.kwargs)
		else:
			return 

	def __call__(self, *args, **kwargs):
		# if self.name in distribution_oracles:  
		# 	return self.evaluator_func(*args, **kwargs)
		# 	#### evaluator for distribution learning, e.g., diversity, validity   
		smiles_lst = args[0]
		if self.name == 'molecule_one_synthesis':
			return self.evaluator_func(*args, **kwargs)

		if type(smiles_lst) == list:

			#### evaluator for single molecule, 
			#### the input of __call__ is a single smiles OR list of smiles
			if isinstance(self.evaluator_func, dict):
				all_ = {}
				for i, fct in self.evaluator_func.items():
					results_lst = []
					for smiles in smiles_lst:
						results_lst.append(fct(smiles, *(args[1:]), **kwargs))
					all_[i] = results_lst
				return all_
			else:
				results_lst = []
				for smiles in smiles_lst:
					results_lst.append(self.evaluator_func(smiles, *(args[1:]), **kwargs))
				return results_lst
		else:	
			## a single smiles
			if type(self.evaluator_func) == dict:
				all_ = {}
				for i, fct in self.evaluator_func.items():
					all_[i] = fct(*args, **kwargs)
				return all_
			else:
				return self.evaluator_func(*args, **kwargs)




