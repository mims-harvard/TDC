import pandas as pd
import numpy as np
import os, sys, json, wget, subprocess
import warnings
warnings.filterwarnings("ignore")

from .. import utils

utils.install("scipy")
from scipy import io

def QM7_process(name, path, target = None):
	if target is None:
		raise AttributeError("Please specify the target label in the target_list.QM7_targets!")

	utils.download_unzip(name, path, 'qm7b.mat')
	df = io.loadmat(os.path.join(path,'qm7b.mat'))

	targets = ["E_PBE0", "E_max_EINDO", "I_max_ZINDO", "HOMO_ZINDO", "LUMO_ZINDO", "E_1st_ZINDO", "IP_ZINDO", "EA_ZINDO", "HOMO_PBE0", "LUMO_PBE0", "HOMO_GW", "LUMO_GW", "alpha_PBE0", "alpha_SCS"]
	targets_index = list(range(len(targets)))
	targets2index = dict(zip(targets, targets_index))	
	y = df['T'].T[targets2index[target]]
	
	drugs = df['X']
	drugs_idx = np.array(['Drug ' + str(i) for i in list(range(drugs.shape[0]))])

	return drugs, y, drugs_idx
