class SmilesCharDictionary(object):
    """
    A fixed dictionary for druglike SMILES.
    Enables smile<->token conversion.

    With a space:0 for padding, Q:1 as the start token and end_of_line \n:2 as the stop token.
    """

    PAD = ' '
    BEGIN = 'Q'
    END = '\n'

    def __init__(self) -> None:
        self.forbidden_symbols = {'Ag', 'Al', 'Am', 'Ar', 'At', 'Au', 'D', 'E', 'Fe', 'G', 'K', 'L', 'M', 'Ra', 'Re',
                                  'Rf', 'Rg', 'Rh', 'Ru', 'T', 'U', 'V', 'W', 'Xe',
                                  'Y', 'Zr', 'a', 'd', 'f', 'g', 'h', 'k', 'm', 'si', 't', 'te', 'u', 'v', 'y'}

        self.char_idx = {self.PAD: 0, self.BEGIN: 1, self.END: 2, '#': 20, '%': 22, '(': 25, ')': 24, '+': 26, '-': 27,
                         '.': 30,
                         '0': 32, '1': 31, '2': 34, '3': 33, '4': 36, '5': 35, '6': 38, '7': 37, '8': 40,
                         '9': 39, '=': 41, 'A': 7, 'B': 11, 'C': 19, 'F': 4, 'H': 6, 'I': 5, 'N': 10,
                         'O': 9, 'P': 12, 'S': 13, 'X': 15, 'Y': 14, 'Z': 3, '[': 16, ']': 18,
                         'b': 21, 'c': 8, 'n': 17, 'o': 29, 'p': 23, 's': 28,
                         "@": 42, "R": 43, '/': 44, "\\": 45, 'E': 46
                         }

        self.idx_char = {v: k for k, v in self.char_idx.items()}

        self.encode_dict = {"Br": 'Y', "Cl": 'X', "Si": 'A', 'Se': 'Z', '@@': 'R', 'se': 'E'}
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

    def allowed(self, smiles) -> bool:
        """
        Determine if smiles string has illegal symbols

        Args:
            smiles: SMILES string

        Returns:
            True if all legal
        """
        for symbol in self.forbidden_symbols:
            if symbol in smiles:
                print('Forbidden symbol {:<2}  in  {}'.format(symbol, smiles))
                return False
        return True

    def encode(self, smiles: str) -> str:
        """
        Replace multi-char tokens with single tokens in SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            sanitized SMILE string with only single-char tokens
        """

        temp_smiles = smiles
        for symbol, token in self.encode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def decode(self, smiles):
        """
        Replace special tokens with their multi-character equivalents.

        Args:
            smiles: SMILES string

        Returns:
            SMILES string with possibly multi-char tokens
        """
        temp_smiles = smiles
        for symbol, token in self.decode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def get_char_num(self) -> int:
        """
        Returns:
            number of characters in the alphabet
        """
        return len(self.idx_char)

    @property
    def begin_idx(self) -> int:
        return self.char_idx[self.BEGIN]

    @property
    def end_idx(self) -> int:
        return self.char_idx[self.END]

    @property
    def pad_idx(self) -> int:
        return self.char_idx[self.PAD]

    def matrix_to_smiles(self, array):
        """
        Converts an matrix of indices into their SMILES representations

        Args:
            array: torch tensor of indices, one molecule per row

        Returns: a list of SMILES, without the termination symbol
        """
        smiles_strings = []

        for row in array:
            predicted_chars = []

            for j in row:
                next_char = self.idx_char[j.item()]
                if next_char == self.END:
                    break
                predicted_chars.append(next_char)

            smi = ''.join(predicted_chars)
            smi = self.decode(smi)
            smiles_strings.append(smi)

        return smiles_strings
