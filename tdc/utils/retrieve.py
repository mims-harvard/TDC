"""Utilities functions for dataset/metadata retrieval
"""
import os, sys
import pandas as pd
from .label_name_list import dataset2target_lists
from .misc import fuzzy_search
from .load import pd_load
from ..metadata import dataset_names, benchmark_names, dataset_list


def get_label_map(
    name,
    path="./data",
    target=None,
    file_format="csv",
    output_format="dict",
    task="DDI",
    name_column="Map",
):
    """to retrieve the biomedical meaning of labels

    Args:
        name (str): the name of the dataset
        path (str, optional): the dataset path, where the data is located
        target (None, optional): the label name
        file_format (str, optional): format of the file
        output_format (str, optional): return a dictionary or a dataframe or the raw array of mapped labels
        task (str, optional): the name of the task
        name_column (str, optional): the name of the column that stores the label name

    Returns:
        dict/pd.DataFrame/np.array: when output_format is dict/df/array

    Raises:
        ValueError: output_format not supported.
    """
    name = fuzzy_search(name, dataset_names[task])
    if target is None:
        target = "Y"
    df = pd_load(name, path)

    if output_format == "dict":
        return dict(zip(df[target].values, df[name_column].values))
    elif output_format == "df":
        return df
    elif output_format == "array":
        return df[name_column].values
    else:
        raise ValueError(
            "Please use the correct output format, select from dict, df, array."
        )


def get_reaction_type(name, path="./data", output_format="array"):
    """to retrieve the type of reactions for reaction dataset

    Args:
        name (str): dataset name
        path (str, optional): dataset path
        output_format (str, optional): output format in dataframe or in raw array format

    Returns:
        pd.DataFrame/np.array: when output_format is df/array

    Raises:
        ValueError: the output format is not supported
    """
    name = fuzzy_search(name, dataset_names["RetroSyn"])
    df = pd_load(name, path)

    if output_format == "df":
        return df
    elif output_format == "array":
        return df["category"].values
    else:
        raise ValueError(
            "Please use the correct output format, select from df, array.")


def retrieve_label_name_list(name):
    """get the set of available labels for query dataset

    Args:
        name (str): rough dataset name

    Returns:
        list: a list of available labels
    """
    name = fuzzy_search(name, dataset_list)
    return dataset2target_lists[name]


def retrieve_dataset_names(name):
    """to get all available dataset names given a task

    Args:
        name (str): the name of query task

    Returns:
        list: a list of available datasets
    """
    return dataset_names[name]


def retrieve_all_benchmarks():
    """to get all available benchmark groups

    Returns:
        list: a list of benchmark group names
    """
    return list(benchmark_names.keys())


def retrieve_benchmark_names(name):
    """to get all available benchmarks given a query benchmark group

    Args:
        name (str): the name of the benchmark group

    Returns:
        list: a list of benchmarks
    """
    name = fuzzy_search(name, list(benchmark_names.keys()))
    datasets = benchmark_names[name]

    dataset_names = []

    for task, datasets in datasets.items():
        for dataset in datasets:
            dataset_names.append(dataset)
    return dataset_names
