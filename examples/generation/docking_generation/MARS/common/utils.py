import os
import random
from rdkit import Chem

from .chem import standardize_smiles

class Recorder():
    def __init__(self, metrics):
        self.metrics = metrics
        self.metric2sum = {}
        self.n_records = 0
        self.reset()

    def reset(self):
        self.n_records = 0
        for metric in self.metrics:
            self.metric2sum[metric] = 0.

    def record(self, n_records, values):
        self.n_records += n_records
        for k, v in zip(self.metrics, values):
            self.metric2sum[k] += v * n_records

    def report_avg(self):
        '''
        @return:
            metric2avg : dictionary from metric names to average values
        '''
        metric2avg = {}
        for metric in self.metrics:
            summ = self.metric2sum[metric]
            avg = summ / self.n_records
            metric2avg[metric] = avg
        return metric2avg
    
    
def print_mols(run_dir, step, mols, scores, dicts):
    path = os.path.join(run_dir, 'mols.txt')
    with open(path, 'a') as f:
        f.write('molecules obtained at step %i\n' % step)
        names = list(dicts[0].keys())
        f.write('score\t%s\tsmiles\n' % '\t'.join(names))
        for i, mol in enumerate(mols):
            try:
                score = scores[i]
                mol = standardize_smiles(mol)
                mol = Chem.RemoveHs(mol)
                smiles = Chem.MolToSmiles(mol)
                target_scores = [dicts[i][name] for name in names]
            except Exception:
                score = 0. 
                smiles = '[INVALID]'
                assert False
                target_scores = [0. for _ in names]
            target_scores = ['%f' % _ for _ in target_scores]
            f.write('%f\t%s\t%s\n' % (score, '\t'.join(target_scores), smiles))

            
def subsample(cnts, r=1e-5, k=.7):
    summ = sum(cnts)
    freq = [1. * c / summ for c in cnts]
    cnts = [min((r / f) ** k, 1.) * c \
        for c, f in zip(cnts, freq)]
    return cnts

def fussy(weights, f=0.):
    if len(weights) == 0:
        return weights
    if isinstance(weights[0], float):
        f = f * sum(weights)
        weights = [w + f for w in weights]
    elif isinstance(weights[0], list):
        for i in range(len(weights)):
            weights[i] = fussy(weights[i], f)
    else: raise NotImplementedError
    return weights

def sample_idx(weights):
    indices = list(range(len(weights)))
    idx = random.choices(indices,
        weights=weights, k=1)[0]
    return idx

