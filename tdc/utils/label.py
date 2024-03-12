"""Utilities functions for transform labels
"""
import numpy as np
import pandas as pd
import os, sys

from .misc import fuzzy_search


def convert_y_unit(y, from_, to_):
    """label unit conversion helper function

    Args:
        y (list): a list of labels
        from_ (str): source units, 'nM'/'p'
        to_ (str): target units, 'p'/'nM'

    Returns:
        np.array: a numpy array of transformed labels
    """
    if from_ == "nM":
        y = y
    elif from_ == "p":
        y = (10**(-y) - 1e-10) / 1e-9
        y = (10**(-y) - 1e-10) / 1e-9

    if to_ == "p":
        y = -np.log10(y * 1e-9 + 1e-10)
    elif to_ == "nM":
        y = y

    return y


def label_transform(y,
                    binary,
                    threshold,
                    convert_to_log,
                    verbose=True,
                    order="descending"):
    """label transformation helper function

    Args:
        y (list): a list of labels
        binary (bool): whether or not to conduct binarization
        threshold (float): the threshold for binarization
        convert_to_log (bool): convert to log-scale for continuous values such as Kd and etc
        verbose (bool, optional): whether or not to print intermediate processing statements
        order (str, optional): if descending, then label is 1 for value less than threshold and vice versus, defaults to 'descending'

    Returns:
        np.array: an array of transformed labels

    Raises:
        ValueError: specify the correct order from 'descending'/'ascending'
    """
    if (len(np.unique(y)) > 2) and binary:
        if verbose:
            print(
                "Binariztion using threshold' + str(threshold) + ', you use specify your threhsold values by threshold = X)",
                flush=True,
                file=sys.stderr,
            )
        if order == "descending":
            y = np.array([1 if i else 0 for i in np.array(y) < threshold])
        elif order == "ascending":
            y = np.array([1 if i else 0 for i in np.array(y) > threshold])
        else:
            raise ValueError(
                "Please select order from 'descending or ascending!")
            raise ValueError(
                "Please select order from 'descending or ascending!")
    else:
        if (len(np.unique(y)) > 2) and convert_to_log:
            if verbose:
                print("To log space...", flush=True, file=sys.stderr)
            y = convert_y_unit(np.array(y), "nM", "p")
        else:
            y = y

    return y


def convert_to_log(y):
    """log conversion helper

    Args:
        y (list): a list of labels

    Returns:
        np.array: an array of log-transformed labels
    """
    y = convert_y_unit(np.array(y), "nM", "p")
    return y


def convert_back_log(y):
    """conversion from log-scale helper

    Args:
        y (list): a list of labels in log-scale

    Returns:
        np.array: an array of nM->p labels
    """
    y = convert_y_unit(np.array(y), "p", "nM")
    return y


def binarize(y, threshold, order="ascending"):
    """binarization of a label list given a pre-specified threshold

    Args:
        y (list): a list of labels
        threshold (float): the threshold for turning label to 1 or 0
        order (str, optional): if order is ascending then for label that is above threshold becomes 1, and below becomes 0, vice versus

    Returns:
        np.array: an array of transformed labels

    Raises:
        AttributeError: select the correct order "ascending/descending"
    """
    if order == "ascending":
        y = np.array([1 if i else 0 for i in np.array(y) > threshold])
    elif order == "descending":
        y = np.array([1 if i else 0 for i in np.array(y) < threshold])
    else:
        raise AttributeError("'order' must be either ascending or descending")
    return y


def label_dist(y, name=None):
    """plot the distribution of label

    Args:
        y (list): a list of labels
        name (None, optional): dataset name
    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except:
        from .misc import install

        install("seaborn")
        install("matplotlib")
        import seaborn as sns
        import matplotlib.pyplot as plt

    median = np.median(y)
    mean = np.mean(y)

    f, (ax_box,
        ax_hist) = plt.subplots(2,
                                sharex=True,
                                gridspec_kw={"height_ratios": (0.15, 1)})
    f, (ax_box,
        ax_hist) = plt.subplots(2,
                                sharex=True,
                                gridspec_kw={"height_ratios": (0.15, 1)})

    if name is None:
        sns.boxplot(y, ax=ax_box).set_title("Label Distribution")
    else:
        sns.boxplot(y, ax=ax_box).set_title("Label Distribution of " +
                                            str(name) + " Dataset")
        sns.boxplot(y, ax=ax_box).set_title("Label Distribution of " +
                                            str(name) + " Dataset")
    ax_box.axvline(median, color="b", linestyle="--")
    ax_box.axvline(mean, color="g", linestyle="--")

    sns.distplot(y, ax=ax_hist)
    ax_hist.axvline(median, color="b", linestyle="--")
    ax_hist.axvline(mean, color="g", linestyle="--")
    ax_hist.legend({"Median": median, "Mean": mean})

    ax_box.set(xlabel="")
    plt.show()
    # print("The median is " + str(median), flush = True, file = sys.stderr)
    # print("The mean is " + str(mean), flush = True, file = sys.stderr)


def NegSample(df, column_names, frac, two_types):
    """Negative Sampling for Binary Interaction Dataset

    Args:
        df (pandas.DataFrame): input dataset dataframe
        column_names (list): column names in the order of [id1, x1, id2, x2]
        frac (float): the ratio of negative samples compared to positive samples
        two_types (bool): whether or not if the two entity types are different (e.g. drug-target) or single entity type (e.g. drug-drug)

    Returns:
        pandas.DataFrame: a new dataframe with negative samples (Y = 0)
    """
    x = int(len(df) * frac)
    id1, x1, id2, x2 = column_names
    df[id1] = df[id1].apply(lambda x: str(x))
    df[id2] = df[id2].apply(lambda x: str(x))

    if not two_types:
        df_unique = np.unique(df[[id1, id2]].values.reshape(-1))
        pos = df[[id1, id2]].values
        pos_set = set([tuple([i[0], i[1]]) for i in pos])
        np.random.seed(1234)
        samples = np.random.choice(df_unique, size=(x, 2), replace=True)
        neg_set = set([tuple([i[0], i[1]]) for i in samples if i[0] != i[1]
                      ]) - pos_set
        neg_set = set([tuple([i[0], i[1]]) for i in samples if i[0] != i[1]
                      ]) - pos_set

        while len(neg_set) < x:
            sample = np.random.choice(df_unique, 2, replace=False)
            sample = tuple([sample[0], sample[1]])
            if sample not in pos_set:
                neg_set.add(sample)
        neg_list = [list(i) for i in neg_set]

        id2seq = dict(df[[id1, x1]].values)
        id2seq.update(df[[id2, x2]].values)

        neg_list_val = []
        for i in neg_list:
            neg_list_val.append([i[0], id2seq[i[0]], i[1], id2seq[i[1]], 0])

        concat_frames = pd.DataFrame(neg_list_val).rename(columns={
            0: id1,
            1: x1,
            2: id2,
            3: x2,
            4: "Y"
        })
        df = pd.concat([df, concat_frames], ignore_index=True, sort=False)
        return df
    else:
        df_unique_id1 = np.unique(df[id1].values.reshape(-1))
        df_unique_id2 = np.unique(df[id2].values.reshape(-1))

        pos = df[[id1, id2]].values
        pos_set = set([tuple([i[0], i[1]]) for i in pos])
        np.random.seed(1234)

        sample_id1 = np.random.choice(df_unique_id1, size=len(df), replace=True)
        sample_id2 = np.random.choice(df_unique_id2, size=len(df), replace=True)

        neg_set = (set([
            tuple([sample_id1[i], sample_id2[i]])
            for i in range(len(df))
            if sample_id1[i] != sample_id2[i]
        ]) - pos_set)
        neg_set = (set([
            tuple([sample_id1[i], sample_id2[i]])
            for i in range(len(df))
            if sample_id1[i] != sample_id2[i]
        ]) - pos_set)

        while len(neg_set) < len(df):
            sample_id1 = np.random.choice(df_unique_id1, size=1, replace=True)
            sample_id2 = np.random.choice(df_unique_id2, size=1, replace=True)

            sample = tuple([sample_id1[0], sample_id2[0]])
            if sample not in pos_set:
                neg_set.add(sample)
        neg_list = [list(i) for i in neg_set]

        id2seq1 = dict(df[[id1, x1]].values)
        id2seq2 = dict(df[[id2, x2]].values)

        neg_list_val = []
        for i in neg_list:
            neg_list_val.append([i[0], id2seq1[i[0]], i[1], id2seq2[i[1]], 0])

        df = pd.concat([
            df,
            pd.DataFrame(neg_list_val).rename(columns={
                0: id1,
                1: x1,
                2: id2,
                3: x2,
                4: "Y"
            })
        ],
                       ignore_index=True,
                       sort=False)
        return df
