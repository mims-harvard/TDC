import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .utils import * 
from .metadata import evaluator_name, distribution_oracles

try:
	from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score, precision_recall_curve
	from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, cohen_kappa_score
except:
	ImportError("Please install sklearn by 'conda install -c anaconda scikit-learn' or 'pip install scikit-learn '! ")

def avg_auc(y_true, y_pred):
	scores = []
	for i in range(np.array(y_true).shape[0]):
	    scores.append(roc_auc_score(y_true[i], y_pred[i]))
	return sum(scores)/len(scores)

def rmse(y_true, y_pred):
	return np.sqrt(mean_squared_error(y_true, y_pred))

def recall_at_precision_k(y_true, y_pred, threshold = 0.9):
	pr, rc, thr = precision_recall_curve(y_true, y_pred)
	if len(np.where(pr >= threshold)[0]) > 0:
		return rc[np.where(pr >= threshold)[0][0]]
	else:
		return 0.

def precision_at_recall_k(y_true, y_pred, threshold = 0.9):
	pr, rc, thr = precision_recall_curve(y_true, y_pred)	 
	if len(np.where(rc >= threshold)[0]) > 0:
		return pr[np.where(rc >= threshold)[0][-1]]
	else:
		return 0.

def pcc(y_true, y_pred):
	return np.corrcoef(y_true, y_pred)[1,0]

class Evaluator:
	def __init__(self, name):
		self.name = fuzzy_search(name, evaluator_name)
		self.assign_evaluator()

	def assign_evaluator(self):
		if self.name == 'roc-auc':
			self.evaluator_func = roc_auc_score 
		elif self.name == 'f1':
			self.evaluator_func = f1_score 
		elif self.name == 'pr-auc':
			self.evaluator_func = average_precision_score 
		elif self.name == 'rp@k':
			self.evaluator_func = recall_at_precision_k
		elif self.name == 'pr@k':
			self.evaluator_func = precision_at_recall_k
		elif self.name == 'precision':
			self.evaluator_func = precision_score
		elif self.name == 'recall':
			self.evaluator_func = recall_score
		elif self.name == 'accuracy':
			self.evaluator_func = accuracy_score
		elif self.name == 'mse':
			self.evaluator_func = mean_squared_error
		elif self.name == 'rmse':
			self.evaluator_func = rmse
		elif self.name == 'mae':
			self.evaluator_func = mean_absolute_error
		elif self.name == 'r2':
			self.evaluator_func = r2_score
		elif self.name == 'pcc':
			self.evaluator_func = pcc
		elif self.name == 'spearman':
			try:
				from scipy import stats
			except:
				ImportError("Please install scipy by 'pip install scipy'! ")
			self.evaluator_func = stats.spearmanr
		elif self.name == 'micro-f1':
			self.evaluator_func = f1_score
		elif self.name == 'macro-f1':
			self.evaluator_func = f1_score
		elif self.name == 'kappa':
			self.evaluator_func = cohen_kappa_score
		elif self.name == 'avg-roc-auc':
			self.evaluator_func = avg_auc
		elif self.name == 'novelty':   	
			from .chem_utils import novelty
			self.evaluator_func = novelty  
		elif self.name == 'diversity':
			from .chem_utils import diversity
			self.evaluator_func = diversity 
		elif self.name == 'validity':
			from .chem_utils import validity
			self.evaluator_func = validity 
		elif self.name == 'uniqueness':
			from .chem_utils import uniqueness
			self.evaluator_func = uniqueness 
		elif self.name == 'kl_divergence':
			from .chem_utils import kl_divergence
			self.evaluator_func = kl_divergence
		elif self.name == 'fcd_distance':
			from .chem_utils import fcd_distance
			self.evaluator_func = fcd_distance

	def __call__(self, *args, **kwargs):
		if self.name in distribution_oracles:  
			return self.evaluator_func(*args, **kwargs)	
		# 	#### evaluator for distribution learning, e.g., diversity, validity   
		y_true = kwargs['y_true'] if 'y_true' in kwargs else args[0]
		y_pred = kwargs['y_pred'] if 'y_pred' in kwargs else args[1]
		if len(args)<=2 and 'threshold' not in kwargs:
			threshold = 0.5 
		else:
			threshold = kwargs['threshold'] if 'threshold' in kwargs else args[2]

		### original __call__(self, y_true, y_pred, threshold = 0.5)
		y_true = np.array(y_true)
		y_pred = np.array(y_pred)
		if self.name in ['precision','recall','f1','accuracy']:
			y_pred = [1 if i > threshold else 0 for i in y_pred]
		if self.name in ['micro-f1', 'macro-f1']:
			return self.evaluator_func(y_true, y_pred, average = self.name[:5])
		if self.name in ['rp@k', 'pr@k']:
			return self.evaluator_func(y_true, y_pred, threshold = threshold)
		if self.name == 'spearman':
			return self.evaluator_func(y_true, y_pred)[0]
		return self.evaluator_func(y_true, y_pred)
