#!/usr/bin/env python
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
    name = op.join(op.dirname(__file__), 'clf_py36.pkl')
    with open(name, "rb") as f:
        clf_model = pickle.load(f)

def get_scores(mols):
    if clf_model is None:
        load_model()
        
    fps = [fingerprints_from_mol(mol) for mol in mols]
    fps = np.concatenate(fps, axis=0)
    scores = clf_model.predict_proba(fps)
    scores = scores[:,1].tolist()
    return scores

def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp