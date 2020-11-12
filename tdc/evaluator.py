import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .utils import * 
from .metadata import evaluator_name
try:
	from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score
	from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, cohen_kappa_score
except:
	ImportError("Please install sklearn by 'conda install -c anaconda scikit-learn' or 'pip install scikit-learn '! ")

def avg_auc(y_true, y_pred):
	scores = []
	for i in range(np.array(y_true).shape[0]):
	    scores.append(roc_auc_score(y_true[i], y_pred[i]))
	return sum(scores)/len(scores)

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
		elif self.name == 'precision':
			self.evaluator_func = precision_score
		elif self.name == 'recall':
			self.evaluator_func = recall_score
		elif self.name == 'accuracy':
			self.evaluator_func = accuracy_score
		elif self.name == 'mse':
			self.evaluator_func = mean_squared_error
		elif self.name == 'mae':
			self.evaluator_func = mean_absolute_error
		elif self.name == 'r2':
			self.evaluator_func = r2_score
		elif self.name == 'micro-f1':
			self.evaluator_func = f1_score
		elif self.name == 'macro-f1':
			self.evaluator_func = f1_score
		elif self.name == 'kappa':
			self.evaluator_func = cohen_kappa_score
		elif self.name == 'avg-roc-auc':
			self.evaluator_func = avg_auc

	def __call__(self, y_true, y_pred, threshold = 0.5):
		y_true = np.array(y_true)
		y_pred = np.array(y_pred)
		if self.name in ['precision','recall','f1','accuracy']:
			y_pred = [1 if i > threshold else 0 for i in y_pred]
		if self.name in ['micro-f1', 'macro-f1']:
			return self.evaluator_func(y_true, y_pred, average = self.name[:5])
		return self.evaluator_func(y_true, y_pred)