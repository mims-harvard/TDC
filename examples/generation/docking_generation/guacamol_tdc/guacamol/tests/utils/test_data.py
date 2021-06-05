import numpy as np
import pytest

from guacamol.utils.data import get_random_subset


def test_subset():
    dataset = list(np.random.rand(100))

    subset = get_random_subset(dataset, 10)

    for s in subset:
        assert s in dataset


def test_subset_if_dataset_too_small():
    dataset = list(np.random.rand(100))

    with pytest.raises(Exception):
        get_random_subset(dataset, 1000)


def test_subset_with_no_seed():
    dataset = list(np.random.rand(100))

    subset1 = get_random_subset(dataset, 10)
    subset2 = get_random_subset(dataset, 10)

    assert subset1 != subset2


def test_subset_with_random_seed():
    dataset = list(np.random.rand(100))

    subset1 = get_random_subset(dataset, 10, seed=33)
    subset2 = get_random_subset(dataset, 10, seed=33)
    subset3 = get_random_subset(dataset, 10, seed=43)

    assert subset1 == subset2
    assert subset1 != subset3
