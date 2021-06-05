from abc import ABCMeta, abstractmethod
from typing import List


class DistributionMatchingGenerator(metaclass=ABCMeta):
    """
    Interface for molecule generators.
    """

    @abstractmethod
    def generate(self, number_samples: int) -> List[str]:
        """
        Samples SMILES strings from a molecule generator.

        Args:
            number_samples: number of molecules to generate

        Returns:
            A list of SMILES strings.
        """
