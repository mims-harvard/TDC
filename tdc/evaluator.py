import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .utils import * 
from .metadata import evaluator_name
try:
	from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score
except:
	ImportError("Please install sklearn by 'conda install -c anaconda scikit-learn' or 'pip install scikit-learn '! ")

class Evaluator:
	def __init__(self, name):
		self.name = fuzzy_search(name, evaluator_name)
		self.assign_evaluator()

	def assign_evaluator(self):
		if  self.name == 'roc-auc':
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


	def __call__(self, y_true, y_pred):
		return self.evaluator_func(y_true, y_pred)