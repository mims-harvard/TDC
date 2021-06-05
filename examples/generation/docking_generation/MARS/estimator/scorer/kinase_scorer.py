import os
import pickle
import numpy as np
from rdkit import Chem, RDLogger

from ...common.chem import fingerprints_from_mol

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


ROOT_DIR = 'MARS/estimator/scorer'
TASKS = ['gsk3b', 'jnk3']
SPLITS = ['val', 'dev']

models = {}
def load_model(task):
    with open(os.path.join(ROOT_DIR, 'kinase_rf/%s.pkl' % task), 'rb') as f:
        models[task] = pickle.load(f, encoding='iso-8859-1')

def get_scores(task, mols):
    model = models.get(task)
    if model is None:
        load_model(task)
        model = models[task]
        
    fps = [fingerprints_from_mol(mol) for mol in mols]
    fps = np.stack(fps, axis=0)
    scores = models[task].predict_proba(fps)
    scores = scores[:,1].tolist()
    return scores

if __name__ == '__main__':
    ### load data
    with open(os.path.join(ROOT_DIR, 'kinase.tsv'), 'r') as f:
        lines = f.readlines()[2:]
        lines = [line.strip('\n').split('\t') for line in lines]
        target = [line[0] for line in lines]
        is_activate = [int(line[1]) for line in lines]
        is_train = [int(line[2]) for line in lines]
        smiles = [line[3] for line in lines]

    data = {}
    for task in TASKS:
        for split in SPLITS:
            subset = '%s_%s' % (task, split)
            data['%s_X' % subset] = []
            data['%s_y' % subset] = []

    smiles_none_cnt = 0
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            smiles_none_cnt += 1
            continue
        fp = fingerprints_from_mol(mol)

        task = target[i] # gsk3b or jnk
        split = SPLITS[is_train[i]]
        subset = '%s_%s' % (task, split)
        data['%s_X' % subset].append(fp)
        data['%s_y' % subset].append(is_activate[i])
    print('invalid smiles count: %i' % smiles_none_cnt)

    ### predict
    for task in TASKS:
        for split in SPLITS:
            subset = '%s_%s' % (task, split)
            X = data['%s_X' % subset]
            y = data['%s_y' % subset]
            X = np.stack(X, axis=0)
            y = np.stack(y, axis=0)
            pred = models[task].predict_proba(X)
            acc = models[task].score(X, y)
            print('accuracy on %s %s: %.4f' % (task, split, acc))
            print(pred)
