import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .utils import fuzzy_search
from .metadata import evaluator_name, distribution_oracles

try:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        average_precision_score,
        precision_score,
        recall_score,
        accuracy_score,
        precision_recall_curve,
    )
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        cohen_kappa_score,
        auc,
        roc_curve,
    )
except:
    ImportError(
        "Please install sklearn by 'conda install -c anaconda scikit-learn' or 'pip install scikit-learn '! "
    )


def avg_auc(y_true, y_pred):
    scores = []
    for i in range(np.array(y_true).shape[0]):
        scores.append(roc_auc_score(y_true[i], y_pred[i]))
    return sum(scores) / len(scores)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def recall_at_precision_k(y_true, y_pred, threshold=0.9):
    pr, rc, thr = precision_recall_curve(y_true, y_pred)
    if len(np.where(pr >= threshold)[0]) > 0:
        return rc[np.where(pr >= threshold)[0][0]]
    else:
        return 0.0


def precision_at_recall_k(y_true, y_pred, threshold=0.9):
    pr, rc, thr = precision_recall_curve(y_true, y_pred)
    if len(np.where(rc >= threshold)[0]) > 0:
        return pr[np.where(rc >= threshold)[0][-1]]
    else:
        return 0.0


def pcc(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[1, 0]


def range_logAUC(true_y, predicted_score, FPR_range=(0.001, 0.1)):
    """
    Author: Yunchao "Lance" Liu (lanceknight26@gmail.com)
    Calculate logAUC in a certain FPR range (default range: [0.001, 0.1]).
    This was used by previous methods [1] and the reason is that only a
    small percentage of samples can be selected for experimental tests in
    consideration of cost. This means only molecules with very high
    predicted score can be worth testing, i.e., the decision
    threshold is high. And the high decision threshold corresponds to the
    left side of the ROC curve, i.e., those FPRs with small values. Also,
    because the threshold cannot be predetermined, the area under the curve
    is used to consolidate all possible thresholds within a certain FPR
    range. Finally, the logarithm is used to bias smaller FPRs. The higher
    the logAUC[0.001, 0.1], the better the performance.

    A perfect classifer gets a logAUC[0.001, 0.1] ) of 1, while a random
    classifer gets a logAUC[0.001, 0.1] ) of around 0.0215 (See [2])

    References:
    [1] Mysinger, M.M. and B.K. Shoichet, Rapid Context-Dependent Ligand
    Desolvation in Molecular Docking. Journal of Chemical Information and
    Modeling, 2010. 50(9): p. 1561-1573.
    [2] Liu, Yunchao, et al. "Interpretable Chirality-Aware Graph Neural
    Network for Quantitative Structure Activity Relationship Modeling in
    Drug Discovery." bioRxiv (2022).
    :param true_y: numpy array of the ground truth. Values are either 0
    (inactive) or 1(active).
    :param predicted_score: numpy array of the predicted score (The
    score does not have to be between 0 and 1)
    :param FPR_range: the range for calculating the logAUC formated in
    (x, y) with x being the lower bound and y being the upper bound
    :return: a numpy array of logAUC of size [1,1]
    """

    # FPR range validity check
    if FPR_range == None:
        raise Exception("FPR range cannot be None")
    lower_bound = FPR_range[0]
    upper_bound = FPR_range[1]
    if lower_bound >= upper_bound:
        raise Exception("FPR upper_bound must be greater than lower_bound")

    fpr, tpr, thresholds = roc_curve(true_y, predicted_score, pos_label=1)

    tpr = np.append(tpr, np.interp([lower_bound, upper_bound], fpr, tpr))
    fpr = np.append(fpr, [lower_bound, upper_bound])

    # Sort both x-, y-coordinates array
    tpr = np.sort(tpr)
    fpr = np.sort(fpr)

    # Get the data points' coordinates. log_fpr is the x coordinate, tpr is the y coordinate.
    log_fpr = np.log10(fpr)
    x = log_fpr
    y = tpr
    lower_bound = np.log10(lower_bound)
    upper_bound = np.log10(upper_bound)

    # Get the index of the lower and upper bounds
    lower_bound_idx = np.where(x == lower_bound)[-1][-1]
    upper_bound_idx = np.where(x == upper_bound)[-1][-1]

    # Create a new array trimmed at the lower and upper bound
    trim_x = x[lower_bound_idx:upper_bound_idx + 1]
    trim_y = y[lower_bound_idx:upper_bound_idx + 1]

    area = auc(trim_x, trim_y) / (upper_bound - lower_bound)
    return area


## code source check here https://github.com/charnley/rmsd


def centroid(X):
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.
    https://en.wikipedia.org/wiki/Centroid
    C = sum(X)/len(X)
    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    C : float
        centroid
    """
    C = X.mean(axis=0)
    return C


def rmsd(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.
    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    diff = np.array(V) - np.array(W)
    N = len(V)
    return np.sqrt((diff * diff).sum() / N)


def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


def kabsch_rmsd(P, Q, W=None, translate=False):
    """
    Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.
    An optional vector of weights W may be provided.
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : array or None
        (N) vector, where N is points.
    translate : bool
        Use centroids to translate vector P and Q unto each other.
    Returns
    -------
    rmsd : float
        root-mean squared deviation
    """

    if translate:
        Q = Q - centroid(Q)
        P = P - centroid(P)

    if W is not None:
        return kabsch_weighted_rmsd(P, Q, W)

    P = kabsch_rotate(P, Q)
    return rmsd(P, Q)


def kabsch_rotate(P, Q):
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated
    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P


def kabsch_weighted_rmsd(P, Q, W=None):
    """
    Calculate the RMSD between P and Q with optional weighhts W
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : vector
        (N) vector, where N is points
    Returns
    -------
    RMSD : float
    """
    R, T, w_rmsd = kabsch_weighted(P, Q, W)
    return w_rmsd


def kabsch_weighted(P, Q, W=None):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q.
    Each vector set is represented as an NxD matrix, where D is the
    dimension of the space.
    An optional vector of weights W may be provided.
    Note that this algorithm does not require that P and Q have already
    been overlayed by a centroid translation.
    The function returns the rotation matrix U, translation vector V,
    and RMS deviation between Q and P', where P' is:
        P' = P * U + V
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : array or None
        (N) vector, where N is points.
    Returns
    -------
    U    : matrix
           Rotation matrix (D,D)
    V    : vector
           Translation vector (D)
    RMSD : float
           Root mean squared deviation between P and Q
    """
    # Computation of the weighted covariance matrix
    CMP = np.zeros(3)
    CMQ = np.zeros(3)
    C = np.zeros((3, 3))
    if W is None:
        W = np.ones(len(P)) / len(P)
    W = np.array([W, W, W]).T
    # NOTE UNUSED psq = 0.0
    # NOTE UNUSED qsq = 0.0
    iw = 3.0 / W.sum()
    n = len(P)
    for i in range(3):
        for j in range(n):
            for k in range(3):
                C[i, k] += P[j, i] * Q[j, k] * W[j, i]
    CMP = (P * W).sum(axis=0)
    CMQ = (Q * W).sum(axis=0)
    PSQ = (P * P * W).sum() - (CMP * CMP).sum() * iw
    QSQ = (Q * Q * W).sum() - (CMQ * CMQ).sum() * iw
    C = (C - np.outer(CMP, CMQ) * iw) * iw

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U, translation vector V, and calculate RMSD:
    U = np.dot(V, W)
    msd = (PSQ + QSQ) * iw - 2.0 * S.sum()
    if msd < 0.0:
        msd = 0.0
    rmsd_ = np.sqrt(msd)
    V = np.zeros(3)
    for i in range(3):
        t = (U[i, :] * CMQ).sum()
        V[i] = CMP[i] - t
    V = V * iw
    return U, V, rmsd_


class Evaluator:
    """evaluator to evaluate predictions

    Args:
            name (str): the name of the evaluator function
    """

    def __init__(self, name):
        """create an evaluate object"""
        self.name = fuzzy_search(name, evaluator_name)
        self.assign_evaluator()

    def assign_evaluator(self):
        """obtain evaluator function given the evaluator name"""
        if self.name == "roc-auc":
            self.evaluator_func = roc_auc_score
        elif self.name == "f1":
            self.evaluator_func = f1_score
        elif self.name == "pr-auc":
            self.evaluator_func = average_precision_score
        elif self.name == "rp@k":
            self.evaluator_func = recall_at_precision_k
        elif self.name == "pr@k":
            self.evaluator_func = precision_at_recall_k
        elif self.name == "precision":
            self.evaluator_func = precision_score
        elif self.name == "recall":
            self.evaluator_func = recall_score
        elif self.name == "accuracy":
            self.evaluator_func = accuracy_score
        elif self.name == "mse":
            self.evaluator_func = mean_squared_error
        elif self.name == "rmse":
            self.evaluator_func = rmse
        elif self.name == "mae":
            self.evaluator_func = mean_absolute_error
        elif self.name == "r2":
            self.evaluator_func = r2_score
        elif self.name == "pcc":
            self.evaluator_func = pcc
        elif self.name == "spearman":
            try:
                from scipy import stats
            except:
                ImportError("Please install scipy by 'pip install scipy'! ")
            self.evaluator_func = stats.spearmanr
        elif self.name == "micro-f1":
            self.evaluator_func = f1_score
        elif self.name == "macro-f1":
            self.evaluator_func = f1_score
        elif self.name == "kappa":
            self.evaluator_func = cohen_kappa_score
        elif self.name == "avg-roc-auc":
            self.evaluator_func = avg_auc
        elif self.name == "novelty":
            from .chem_utils import novelty

            self.evaluator_func = novelty
        elif self.name == "diversity":
            from .chem_utils import diversity

            self.evaluator_func = diversity
        elif self.name == "validity":
            from .chem_utils import validity

            self.evaluator_func = validity
        elif self.name == "uniqueness":
            from .chem_utils import uniqueness

            self.evaluator_func = uniqueness
        elif self.name == "kl_divergence":
            from .chem_utils import kl_divergence

            self.evaluator_func = kl_divergence
        elif self.name == "fcd_distance":
            from .chem_utils import fcd_distance

            self.evaluator_func = fcd_distance
        elif self.name == "range_logAUC":
            self.evaluator_func = range_logAUC
        elif self.name == "rmsd":
            self.evaluator_func = rmsd
        elif self.name == "kabsch_rmsd":
            self.evaluator_func = kabsch_rmsd

    def __call__(self, *args, **kwargs):
        """call the evaluator function on targets and predictions

        Args:
            *args: targets, predictions, and other information
            **kwargs: other auxilliary inputs for some evaluators

        Returns:
            float: the evaluator output
        """
        if self.name in distribution_oracles:
            return self.evaluator_func(*args, **kwargs)
        # 	#### evaluator for distribution learning, e.g., diversity, validity
        y_true = kwargs["y_true"] if "y_true" in kwargs else args[0]
        y_pred = kwargs["y_pred"] if "y_pred" in kwargs else args[1]
        if len(args) <= 2 and "threshold" not in kwargs:
            threshold = 0.5
        else:
            threshold = kwargs["threshold"] if "threshold" in kwargs else args[2]

        ### original __call__(self, y_true, y_pred, threshold = 0.5)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if self.name in ["precision", "recall", "f1", "accuracy"]:
            y_pred = [1 if i > threshold else 0 for i in y_pred]
        if self.name in ["micro-f1", "macro-f1"]:
            return self.evaluator_func(y_true, y_pred, average=self.name[:5])
        if self.name in ["rp@k", "pr@k"]:
            return self.evaluator_func(y_true, y_pred, threshold=threshold)
        if self.name == "spearman":
            return self.evaluator_func(y_true, y_pred)[0]
        return self.evaluator_func(y_true, y_pred)
