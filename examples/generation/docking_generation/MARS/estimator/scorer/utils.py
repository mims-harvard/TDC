import numpy as np
from rdkit.Chem import AllChem
from rdkit import DataStructs

def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 1024
    nfp = np.zeros((size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[nidx] += int(v)
    return nfp