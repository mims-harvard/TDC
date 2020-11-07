import pandas as pd
import numpy as np
import os, sys, json, wget, subprocess
import warnings
warnings.filterwarnings("ignore")

from .. import utils

utils.install("scipy")
from scipy import io

def QM7_process(name, path, target = None):
	from scipy import io
	import os
	import numpy as np

	inx = io.loadmat(os.path.join('/Users/kexinhuang/Downloads/qm7b','qm7b.mat'))

	targets = ["E_PBE0", "E_max_EINDO", "I_max_ZINDO", "HOMO_ZINDO", "LUMO_ZINDO", "E_1st_ZINDO", "IP_ZINDO", "EA_ZINDO", "HOMO_PBE0", "LUMO_PBE0", "HOMO_GW", "LUMO_GW", "alpha_PBE0", "alpha_SCS"]
	import pandas as pd
	df = pd.DataFrame()
	df['X'] = pd.Series([i for i in inx['X']])
	df = pd.concat([df, pd.DataFrame(inx['T'], columns = targets)], axis = 1)
	df['ID'] = ['Drug ' + str(i+1) for i in range(len(df))]
	df.to_pickle('/Users/kexinhuang/Desktop/qm7b.pkl')

def QM8_9():
	import deepchem as dc
	d = dc.molnet.load_qm9()
	import numpy as np
	X = np.concatenate([d[1][0].X, d[1][1].X, d[1][2].X])
	Y = np.concatenate([d[1][0].y, d[1][1].y, d[1][2].y])

	targets = d[0]

	import pandas as pd
	df = pd.DataFrame()
	df['X'] = pd.Series([i for i in X])
	df = pd.concat([df, pd.DataFrame(Y, columns = targets)], axis = 1)
	df['ID'] = ['Drug ' + str(i+1) for i in range(len(df))]
	df.to_pickle('/Users/kexinhuang/Desktop/qm9.pkl')