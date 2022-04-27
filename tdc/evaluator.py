import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from .utils import fuzzy_search 
from .metadata import evaluator_name, distribution_oracles

try:
	from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score, precision_recall_curve
	from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, cohen_kappa_score, auc, roc_curve
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

def range_logAUC(true_y, predicted_score, FPR_range=(0.001, 0.1)):
    """
    Author: Yunchao "Lance" Liu (lanceknight26@gmail.com)
    Calculate logAUC in a certain FPR range (default range: [0.001, 0.1]).
    This was used by previous methods [1] and the reason is that only a
    small percentage of samples can be selected for experimental tests in
    consideration of cost. This means only molecules with very high
    predicted score can be worth testing, i.e., the decision
    threshold is high. And the high decision threshold corresponds to the
    left side of the ROC curve, i.e., those FPRs with small values. Also,
    because the threshold cannot be predetermined, the area under the curve
    is used to consolidate all possible thresholds within a certain FPR
    range. Finally, the logarithm is used to bias smaller FPRs. The higher
    the logAUC[0.001, 0.1], the better the performance.

    A perfect classifer gets a logAUC[0.001, 0.1] ) of 1, while a random
    classifer gets a logAUC[0.001, 0.1] ) of around 0.0215 (See [2])

    References:
    [1] Mysinger, M.M. and B.K. Shoichet, Rapid Context-Dependent Ligand
    Desolvation in Molecular Docking. Journal of Chemical Information and
    Modeling, 2010. 50(9): p. 1561-1573.
    [2] Mendenhall, J. and J. Meiler, Improving quantitative
    structure–activity relationship models using Artificial Neural Networks
    trained with dropout. Journal of computer-aided molecular design,
    2016. 30(2): p. 177-189.
    :param true_y: numpy array of the ground truth. Values are either 0 (
    inactive) or 1(active).
    :param predicted_score: numpy array of the predicted score (The
    score does not have to be between 0 and 1)
    :param FPR_range: the range for calculating the logAUC formated in
    (x, y) with x being the lower bound and y being the upper bound
    :return: a numpy array of logAUC of size [1,1]
    """

    # FPR range validity check
    if FPR_range == None:
        raise Exception('FPR range cannot be None')
    lower_bound = np.log10(FPR_range[0])
    upper_bound = np.log10(FPR_range[1])
    if (lower_bound >= upper_bound):
        raise Exception('FPR upper_bound must be greater than lower_bound')

    # Get the data points' coordinates. log_fpr is the x coordinate, tpr is
    # the y coordinate.
    fpr, tpr, thresholds = roc_curve(true_y, predicted_score, pos_label=1)
    log_fpr = np.log10(fpr)

    # Intecept the curve at the two ends of the region, i.e., lower_bound,
    # and upper_bound
    tpr = np.append(tpr, np.interp([lower_bound, upper_bound], log_fpr, tpr))
    log_fpr = np.append(log_fpr, [lower_bound, upper_bound])

    # Sort both x-, y-coordinates array
    x = np.sort(log_fpr)
    y = np.sort(tpr)

    # For visulization of the plot before trimming, uncomment the following
    # line, with proper libray imported
    # plt.plot(x, y)
    # For visulization of the plot in the trimmed area, uncomment the
    # following line
    # plt.xlim(lower_bound, upper_bound)

    # Get the index of the lower and upper bounds
    lower_bound_idx = np.where(x == lower_bound)[-1][-1]
    upper_bound_idx = np.where(x == upper_bound)[-1][-1]

    # Create a new array trimmed at the lower and upper bound
    trim_x = x[lower_bound_idx:upper_bound_idx + 1]
    trim_y = y[lower_bound_idx:upper_bound_idx + 1]

    area = auc(trim_x, trim_y) / 2

    return area

class Evaluator:

	"""evaluator to evaluate predictions
	
	Args:
		name (str): the name of the evaluator function
	"""
	
	def __init__(self, name):
		"""create an evaluate object
		"""
		self.name = fuzzy_search(name, evaluator_name)
		self.assign_evaluator()

	def assign_evaluator(self):
		"""obtain evaluator function given the evaluator name
		"""
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
		elif self.name == 'range_logAUC':
			self.evaluator_func = range_logAUC

	def __call__(self, *args, **kwargs):
		"""call the evaluator function on targets and predictions
		
		Args:
		    *args: targets, predictions, and other information
		    **kwargs: other auxilliary inputs for some evaluators
		
		Returns:
		    float: the evaluator output
		"""
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
