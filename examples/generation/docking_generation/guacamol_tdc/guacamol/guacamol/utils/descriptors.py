from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, rdMolDescriptors


def logP(mol: Mol) -> float:
    return Descriptors.MolLogP(mol)


def qed(mol: Mol) -> float:
    return Descriptors.qed(mol)


def tpsa(mol: Mol) -> float:
    return Descriptors.TPSA(mol)


def bertz(mol: Mol) -> float:
    return Descriptors.BertzCT(mol)


def mol_weight(mol: Mol) -> float:
    return Descriptors.MolWt(mol)


def num_H_donors(mol: Mol) -> int:
    return Descriptors.NumHDonors(mol)


def num_H_acceptors(mol: Mol) -> int:
    return Descriptors.NumHAcceptors(mol)


def num_rotatable_bonds(mol: Mol) -> int:
    return Descriptors.NumRotatableBonds(mol)


def num_rings(mol: Mol) -> int:
    return rdMolDescriptors.CalcNumRings(mol)


def num_aromatic_rings(mol: Mol) -> int:
    return rdMolDescriptors.CalcNumAromaticRings(mol)


def num_atoms(mol: Mol) -> int:
    """
    Returns the total number of atoms, H included
    """
    mol = Chem.AddHs(mol)
    return mol.GetNumAtoms()


class AtomCounter:

    def __init__(self, element: str) -> None:
        """
        Args:
            element: element to count within a molecule
        """
        self.element = element

    def __call__(self, mol: Mol) -> int:
        """
        Count the number of atoms of a given type.

        Args:
            mol: molecule

        Returns:
            The number of atoms of the given type.
        """
        # if the molecule contains H atoms, they may be implicit, so add them
        if self.element == 'H':
            mol = Chem.AddHs(mol)

        return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == self.element)
