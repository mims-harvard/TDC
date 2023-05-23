"""miscellaneous utilities functions
"""
import os, sys
import numpy as np
import pandas as pd
import subprocess
import pickle
from fuzzywuzzy import fuzz
from torch_geometric.data import Data
import h5py
import torch 

def fuzzy_search(name, dataset_names):
    """fuzzy matching between the real dataset name and the input name

    Args:
        name (str): input dataset name given by users
        dataset_names (str): the exact dataset name in TDC

    Returns:
        s: the real dataset name

    Raises:
        ValueError: the wrong task name, no name is matched
    """
    name = name.lower()
    if name[:4] == "tdc.":
        name = name[4:]
    if name in dataset_names:
        s = name
    else:
        # print("========fuzzysearch=======", dataset_names, name)
        s = get_closet_match(dataset_names, name)[0]
    if s in dataset_names:
        return s
    else:
        raise ValueError(
            s + " does not belong to this task, please refer to the correct task name!"
        )


def get_closet_match(predefined_tokens, test_token, threshold=0.8):
    """Get the closest match by Levenshtein Distance.

    Args:
        predefined_tokens (list): Predefined string tokens.
        test_token (str): User input that needs matching to existing tokens.
        threshold (float, optional): The lowest match score to raise errors, defaults to 0.8

    Returns:
        str: the exact token with highest matching prob
            float: probability

    Raises:
        ValueError: no name is matched
    """
    prob_list = []

    for token in predefined_tokens:
        # print(token)
        prob_list.append(fuzz.ratio(str(token).lower(), str(test_token).lower()))

    assert len(prob_list) == len(predefined_tokens)

    prob_max = np.nanmax(prob_list)
    token_max = predefined_tokens[np.nanargmax(prob_list)]

    # match similarity is low
    if prob_max / 100 < threshold:
        print_sys(predefined_tokens)
        raise ValueError(
            test_token, "does not match to available values. " "Please double check."
        )
    return token_max, prob_max / 100


def save_dict(path, obj):
    """save an object to a pickle file

    Args:
        path (str): the path to save the pickle file
        obj (object): any file
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(path):
    """load an object from a path

    Args:
        path (str): the path where the pickle file locates

    Returns:
        object: loaded pickle file
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def install(package):
    """install pip package

    Args:
        package (str): package name
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush=True, file=sys.stderr)


def to_submission_format(results):
    """convert the results to submission-ready format in leaderboard

    Args:
        results (dict): a dictionary of metrics across five runs

    Returns:
        dict: a dictionary of metrics and values with mean and std
    """
    df = pd.DataFrame(results)

    def get_metric(x):
        metric = []
        for i in x:
            metric.append(list(i.values())[0])
        return [round(np.mean(metric), 3), round(np.std(metric), 3)]

    return dict(df.apply(get_metric, axis=1))




def h5py_to_pyg(self, h5_file_path):
    h5File = h5py.File(h5_file_path, "r")
    data = Data()

    amino_types = h5File['amino_types'][()]
    atom_names = h5File['atom_names'][()] 
    atom_amino_id = h5File['atom_amino_id'][()] 
    atom_pos = h5File['atom_pos'][()][0] 

    # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
    mask_n = np.char.equal(atom_names, b'N')
    mask_ca = np.char.equal(atom_names, b'CA')
    mask_c = np.char.equal(atom_names, b'C')
    mask_cb = np.char.equal(atom_names, b'CB')
    mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
    mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
    mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
    mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
    mask_h = np.char.equal(atom_names, b'NH1')

    pos_n = np.full((len(amino_types),3),np.nan)
    pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
    pos_n = torch.FloatTensor(pos_n)

    pos_ca = np.full((len(amino_types),3),np.nan)
    pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
    pos_ca = torch.FloatTensor(pos_ca)

    pos_c = np.full((len(amino_types),3),np.nan)
    pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
    pos_c = torch.FloatTensor(pos_c)

    # if data only contain pos_ca, we set the position of C and N as the position of CA
    pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
    pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

    pos_cb = np.full((len(amino_types),3),np.nan)
    pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
    pos_cb = torch.FloatTensor(pos_cb)

    pos_g = np.full((len(amino_types),3),np.nan)
    pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
    pos_g = torch.FloatTensor(pos_g)

    pos_d = np.full((len(amino_types),3),np.nan)
    pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
    pos_d = torch.FloatTensor(pos_d)

    pos_e = np.full((len(amino_types),3),np.nan)
    pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
    pos_e = torch.FloatTensor(pos_e)

    pos_z = np.full((len(amino_types),3),np.nan)
    pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
    pos_z = torch.FloatTensor(pos_z)

    pos_h = np.full((len(amino_types),3),np.nan)
    pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
    pos_h = torch.FloatTensor(pos_h)

    data.x = torch.unsqueeze(torch.tensor(amino_types),1)
    data.coords_ca = pos_ca
    data.coords_n = pos_n
    data.coords_c = pos_c

    h5File.close()
    return data