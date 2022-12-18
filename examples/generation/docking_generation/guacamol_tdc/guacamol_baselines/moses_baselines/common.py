from typing import List


def read_smiles(smiles_file: str) -> List[str]:
    with open(smiles_file, "r") as f:
        smiles_list = [line.strip() for line in f.readlines()]
    return smiles_list
