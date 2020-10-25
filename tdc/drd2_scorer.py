import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn import svm
import pickle
import re
import os.path as op
rdBase.DisableLog('rdApp.error')

"""Scores based on an ECFP classifier for activity."""

clf_model = None
def load_model():
    global clf_model
    # name = op.join(op.dirname(__file__), 'clf_py36.pkl')
    name = 'tdc/clf_py36.pkl'
    with open(name, "rb") as f:
        clf_model = pickle.load(f)

def get_score(smile):
    if clf_model is None:
        load_model()

    mol = Chem.MolFromSmiles(smile)
    if mol:
        fp = fingerprints_from_mol(mol)
        score = clf_model.predict_proba(fp)[:, 1]
        return float(score)
    return 0.0

def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp





if __name__ == "__main__":
    smiles = '[H][C@@]12C[C@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C'
    print(get_score(smiles))    



