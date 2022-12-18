class ChemblFileReader:
    """
    This class can repeatedly generate an iterator for iterating over the content of a file containing SMILES strings.
    """

    def __init__(self, smiles_file_path: str):
        """
        Args:
            smiles_file_path: Path of a file containing a list of SMILES strings.
        """
        self.smiles_file_path = smiles_file_path

    def __iter__(self):
        with open(self.smiles_file_path) as f:
            for line in f:
                yield line.strip()
