from typing import List

import numpy as np


def arithmetic_mean(values: List[float]) -> float:
    """
    Computes the arithmetic mean of a list of values.
    """
    return sum(values) / len(values)


def geometric_mean(values: List[float]) -> float:
    """
    Computes the geometric mean of a list of values.
    """
    a = np.array(values)
    return a.prod() ** (1.0 / len(a))
