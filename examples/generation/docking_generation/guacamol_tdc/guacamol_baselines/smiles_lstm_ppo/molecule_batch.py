from typing import List

from guacamol.utils.chemistry import canonicalize
from guacamol.utils.data import remove_duplicates


class MoleculeBatch(object):
    """
    Delivers useful properties about a batch of generated SMILES strings.

    Canonicalization of the SMILES strings, and removal of the duplicates, will be
    done only one time, and only if necessary.
    """

    def __init__(self, smiles: List[str]) -> None:
        self._smiles = smiles
        self._canonical_smiles = None
        self._unique_canonical_smiles = None

    @property
    def canonical_smiles(self):
        self._canonicalize()
        return self._canonical_smiles

    @property
    def unique_canonical_smiles(self):
        self._remove_duplicates()
        return self._unique_canonical_smiles

    @property
    def size(self):
        return len(self._smiles)

    @property
    def number_valid(self):
        self._canonicalize()
        return len(self._canonical_smiles)

    @property
    def number_unique(self):
        self._remove_duplicates()
        return len(self._unique_canonical_smiles)

    @property
    def ratio_valid(self):
        return self.number_valid / self.size

    @property
    def ratio_unique(self):
        """The ratio of unique valid molecules compared to the total size"""
        return self.number_unique / self.size

    @property
    def ratio_unique_among_valid(self):
        return self.number_unique / self.number_valid

    def _canonicalize(self):
        if self._canonical_smiles is not None:
            return

        canonical = [canonicalize(mol) for mol in self._smiles]
        self._canonical_smiles = [s for s in canonical if s is not None]

    def _remove_duplicates(self):
        if self._unique_canonical_smiles is not None:
            return

        self._canonicalize()
        self._unique_canonical_smiles = remove_duplicates(self._canonical_smiles)
