"""Utilities functions for splitting dataset 
"""
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from .misc import print_sys


def create_fold(df, fold_seed, frac):
    """create random split

    Args:
        df (pd.DataFrame): dataset dataframe
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac=val_frac / (1 - test_frac),
                           replace=False,
                           random_state=1)
    train = train_val[~train_val.index.isin(val.index)]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def create_fold_setting_cold(df, fold_seed, frac, entities):
    """create cold-split where given one or multiple columns, it first splits based on
    entities in the columns and then maps all associated data points to the partition

    Args:
            df (pd.DataFrame): dataset dataframe
            fold_seed (int): the random seed
            frac (list): a list of train/valid/test fractions
            entities (Union[str, List[str]]): either a single "cold" entity or a list of
                    "cold" entities on which the split is done

    Returns:
            dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    # For each entity, sample the instances belonging to the test datasets
    test_entity_instances = [
        df[e].drop_duplicates().sample(frac=test_frac,
                                       replace=False,
                                       random_state=fold_seed).values
        for e in entities
    ]

    # Select samples where all entities are in the test set
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            "No test samples found. Try another seed, increasing the test frac or a "
            "less stringent splitting strategy.")

    # Proceed with validation data
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e].drop_duplicates().sample(frac=val_frac / (1 - test_frac),
                                              replace=False,
                                              random_state=fold_seed).values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            "No validation samples found. Try another seed, increasing the test frac "
            "or a less stringent splitting strategy.")

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return {
        "train": train.reset_index(drop=True),
        "valid": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def create_scaffold_split(df, seed, frac, entity):
    """create scaffold split. it first generates molecular scaffold for each molecule and then split based on scaffolds
    reference: https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py

    Args:
        df (pd.DataFrame): dataset dataframe
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions
        entity (str): the column name for where molecule stores

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except:
        raise ImportError(
            "Please install rdkit by 'conda install -c conda-forge rdkit'! ")
    from tqdm import tqdm
    from random import Random

    from collections import defaultdict

    random = Random(seed)

    s = df[entity].values
    scaffolds = defaultdict(set)
    idx2mol = dict(zip(list(range(len(s))), s))

    error_smiles = 0
    for i, smiles in tqdm(enumerate(s), total=len(s)):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smiles), includeChirality=False)
            scaffolds[scaffold].add(i)
        except:
            print_sys(smiles + " returns RDKit error and is thus omitted...")
            error_smiles += 1

    train, val, test = [], [], []
    train_size = int((len(df) - error_smiles) * frac[0])
    val_size = int((len(df) - error_smiles) * frac[1])
    test_size = (len(df) - error_smiles) - train_size - val_size
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # index_sets = sorted(list(scaffolds.values()), key=lambda i: len(i), reverse=True)
    index_sets = list(scaffolds.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    if frac[2] == 0:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            else:
                val += index_set
                val_scaffold_count += 1
    else:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

    return {
        "train": df.iloc[train].reset_index(drop=True),
        "valid": df.iloc[val].reset_index(drop=True),
        "test": df.iloc[test].reset_index(drop=True),
    }


def create_combination_generation_split(dict1, dict2, seed, frac):
    """create random split

    Args:
        dict: data dict
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    train_frac, val_frac, test_frac = frac
    length = len(dict1["coord"])
    indices = np.random.permutation(length)
    train_idx, val_idx, test_idx = (
        indices[:int(length * train_frac)],
        indices[int(length * train_frac):int(length * (train_frac + val_frac))],
        indices[int(length * (train_frac + val_frac)):],
    )

    return {
        "train": {
            "protein_coord": [dict1["coord"][i] for i in train_idx],
            "protein_atom_type": [dict1["atom_type"][i] for i in train_idx],
            "ligand_coord": [dict2["coord"][i] for i in train_idx],
            "ligand_atom_type": [dict2["atom_type"][i] for i in train_idx],
        },
        "valid": {
            "protein_coord": [dict1["coord"][i] for i in val_idx],
            "protein_atom_type": [dict1["atom_type"][i] for i in val_idx],
            "ligand_coord": [dict2["coord"][i] for i in val_idx],
            "ligand_atom_type": [dict2["atom_type"][i] for i in val_idx],
        },
        "test": {
            "protein_coord": [dict1["coord"][i] for i in test_idx],
            "protein_atom_type": [dict1["atom_type"][i] for i in test_idx],
            "ligand_coord": [dict2["coord"][i] for i in test_idx],
            "ligand_atom_type": [dict2["atom_type"][i] for i in test_idx],
        },
    }


def create_combination_split(df, seed, frac):
    """
    Function for splitting drug combination dataset such that no combinations are shared across the split

    Args:
        df (pd.Dataframe): dataset to split
        seed (int): random seed
        frac (list): split fraction as a list

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """

    test_size = int(len(df) * frac[2])
    train_size = int(len(df) * frac[0])
    val_size = len(df) - train_size - test_size
    np.random.seed(seed)

    # Create a new column for combination names
    df["concat"] = df["Drug1_ID"] + "," + df["Drug2_ID"]

    # Identify shared drug combinations across all target classes
    combinations = []
    for c in df["Cell_Line_ID"].unique():
        df_cell = df[df["Cell_Line_ID"] == c]
        combinations.append(set(df_cell["concat"].values))

    intxn = combinations[0]
    for c in combinations:
        intxn = intxn.intersection(c)

    # Split combinations into train, val and test
    test_choices = np.random.choice(list(intxn),
                                    int(test_size /
                                        len(df["Cell_Line_ID"].unique())),
                                    replace=False)
    trainval_intxn = intxn.difference(test_choices)
    val_choices = np.random.choice(
        list(trainval_intxn),
        int(val_size / len(df["Cell_Line_ID"].unique())),
        replace=False,
    )

    ## Create train and test set
    test_set = df[df["concat"].isin(test_choices)].drop(columns=["concat"])
    val_set = df[df["concat"].isin(val_choices)]
    train_set = df[~df["concat"].isin(test_choices)].reset_index(drop=True)
    train_set = train_set[~train_set["concat"].isin(val_choices)]

    return {
        "train": train_set.reset_index(drop=True),
        "valid": val_set.reset_index(drop=True),
        "test": test_set.reset_index(drop=True),
    }


# create time split


def create_fold_time(df, frac, date_column):
    """create splits based on time

    Args:
        df (pd.DataFrame): the dataset dataframe
        frac (list): list of train/valid/test fractions
        date_column (str): the name of the column that contains the time info

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    df = df.sort_values(by=date_column).reset_index(drop=True)
    train_frac, val_frac, test_frac = frac[0], frac[1], frac[2]

    split_date = df[:int(len(df) *
                         (train_frac + val_frac))].iloc[-1][date_column]
    test = df[df[date_column] >= split_date].reset_index(drop=True)
    train_val = df[df[date_column] < split_date]

    split_date_valid = train_val[:int(
        len(train_val) * train_frac /
        (train_frac + val_frac))].iloc[-1][date_column]
    train = train_val[train_val[date_column] <= split_date_valid].reset_index(
        drop=True)
    valid = train_val[train_val[date_column] > split_date_valid].reset_index(
        drop=True)

    return {
        "train": train,
        "valid": valid,
        "test": test,
        "split_time": {
            "train_time_frame": (df.iloc[0][date_column], split_date_valid),
            "valid_time_frame": (split_date_valid, split_date),
            "test_time_frame": (split_date, df.iloc[-1][date_column]),
        },
    }


def create_group_split(train_val, seed, holdout_frac, group_column):
    """split within each stratification defined by the group column for training/validation split

    Args:
        train_val (pd.DataFrame): the train+valid dataframe to split on
        seed (int): the random seed
        holdout_frac (float): the fraction of validation
        group_column (str): the name of the group column

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    for i in train_val[group_column].unique():
        train_val_temp = train_val[train_val[group_column] == i]
        np.random.seed(seed)
        msk = np.random.rand(len(train_val_temp)) < (1 - holdout_frac)
        train_df = pd.concat([train_df, train_val_temp[msk]], axis=0)
        val_df = pd.concat([val_df, train_val_temp[~msk]], axis=0)

    return {
        "train": train_df.reset_index(drop=True),
        "valid": val_df.reset_index(drop=True),
    }
