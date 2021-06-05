from typing import List

import numpy as np
from guacamol.distribution_matching_generator import DistributionMatchingGenerator


class RandomSmilesSampler(DistributionMatchingGenerator):
    """
    Generator that samples SMILES strings from a predefined list.
    """

    def __init__(self, molecules: List[str]) -> None:
        """
        Args:
            molecules: list of molecules from which the samples will be drawn
        """
        self.molecules = molecules

    def generate(self, number_samples: int) -> List[str]:
        return list(np.random.choice(self.molecules, size=number_samples))
