import os
import csv
import pickle
from rdkit import Chem


def load_mols(data_dir, mols_file):
    path = os.path.join(data_dir, mols_file)
    if mols_file.endswith('.smiles'): # processed zinc data
        mols = []
        with open(path, 'r') as f:
            lines = f.readlines(int(1e8))
            lines = [line.strip('\n').split('\t') for line in lines]
            smiles = [line[1] for line in lines]
    elif mols_file.endswith('.csv'): # zinc_250k.csv
        reader = csv.reader(open(path, 'r'))
        smiles = [line[0].strip() for line in reader]
        smiles = smiles[1:]
    elif mols_file.endswith('.txt'): # chembl.txt
        with open(path, 'r') as f:
            lines = f.readlines()
            smiles = [line.strip() for line in lines]
    elif mols_file == 'kinase.tsv': # kinase data
        with open(path, 'r') as f:
            lines = f.readlines()[2:]
            lines = [line.strip().split('\t') for line in lines]
            lines = [line for line in lines if line[1] == '1']
            smiles = [line[-1] for line in lines]
    elif mols_file.startswith('actives') or \
        mols_file.startswith('init'): # refenrence active mols
        with open(path, 'r') as f:
            lines = f.readlines()[1:]
            lines = [line.strip().split(',') for line in lines]
            smiles = [line[0] for line in lines]
    else: raise NotImplementedError
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = [mol for mol in mols if mol]
    print('loaded %i molecules' % len(mols))
    return mols


class Vocab():
    def __init__(self, arms, cnts, smiles2idx):
        self.arms = arms
        self.cnts = cnts
        self.smiles2idx = smiles2idx

def load_vocab(data_dir, profix, vocab_size=200):
    vocab_dir = os.path.join(data_dir, 'vocab_%s' % profix)
    with open(os.path.join(vocab_dir, 'arms.pkl'), 'rb') as f:
        arms = pickle.load(f)
        arms = arms[:vocab_size]
    with open(os.path.join(vocab_dir, 'arms.smiles'), 'r') as f:
        lines = f.readlines()
        lines = lines[:vocab_size]
        lines = [line.strip('\n').split('\t') for line in lines]
        cnts = [float(line[0]) for line in lines]
        smiles_list = [line[1] for line in lines][:10]
        smiles2idx = {smiles : i for i, smiles in enumerate(smiles_list)}
    print('loaded vocab of size %i' % len(arms))
    vocab = Vocab(arms, cnts, smiles2idx)
    return vocab
