from typing import List, Set

from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.utils.chemistry import is_valid, canonicalize


def sample_valid_molecules(model: DistributionMatchingGenerator, number_molecules: int, max_tries=10) -> List[str]:
    """
    Sample from the given generator until the desired number of valid molecules
    has been sampled (i.e., ignore invalid molecules).

    Args:
        model: model to sample from
        number_molecules: number of valid molecules to generate
        max_tries: determines the maximum number N of samples to draw, N = number_molecules * max_tries

    Returns:
        A list of number_molecules valid molecules. If this was not possible with the given max_tries, the list may be shorter.
    """

    max_samples = max_tries * number_molecules
    number_already_sampled = 0

    valid_molecules: List[str] = []

    while len(valid_molecules) < number_molecules and number_already_sampled < max_samples:
        remaining_to_sample = number_molecules - len(valid_molecules)

        samples = model.generate(remaining_to_sample)
        number_already_sampled += remaining_to_sample

        valid_molecules += [m for m in samples if is_valid(m)]

    return valid_molecules


def sample_unique_molecules(model: DistributionMatchingGenerator, number_molecules: int, max_tries=10) -> List[str]:
    """
    Sample from the given generator until the desired number of unique (distinct) molecules
    has been sampled (i.e., ignore duplicate molecules).

    Args:
        model: model to sample from
        number_molecules: number of unique (distinct) molecules to generate
        max_tries: determines the maximum number N of samples to draw, N = number_molecules * max_tries

    Returns:
        A list of number_molecules unique molecules, in canonalized form.
        If this was not possible with the given max_tries, the list may be shorter.
        The generation order is kept.
    """

    max_samples = max_tries * number_molecules
    number_already_sampled = 0

    unique_list: List[str] = []
    unique_set: Set[str] = set()

    while len(unique_list) < number_molecules and number_already_sampled < max_samples:
        remaining_to_sample = number_molecules - len(unique_list)

        samples = model.generate(remaining_to_sample)
        number_already_sampled += remaining_to_sample

        for smiles in samples:
            canonical_smiles = canonicalize(smiles)
            if canonical_smiles is not None and canonical_smiles not in unique_set:
                unique_set.add(canonical_smiles)
                unique_list.append(canonical_smiles)

    # this should always be True
    assert len(unique_set) == len(unique_list)

    return unique_list
