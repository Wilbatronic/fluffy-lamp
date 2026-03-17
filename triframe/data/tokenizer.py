"""DNA and amino acid tokenizers."""


class DNATokenizer:
    """Maps DNA characters to integer IDs. A=0, C=1, G=2, T=3, N=4. Case-insensitive."""

    _CHAR_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    _ID_TO_CHAR = {v: k for k, v in _CHAR_TO_ID.items()}
    VOCAB_SIZE = 5

    def encode(self, sequence: str) -> list[int]:
        """Encode a DNA string to a list of integer IDs."""
        return [self._CHAR_TO_ID.get(c, 4) for c in sequence.upper()]

    def decode(self, ids: list[int]) -> str:
        """Decode integer IDs back to a DNA string."""
        return "".join(self._ID_TO_CHAR.get(i, "N") for i in ids)


class AATokenizer:
    """Maps amino acid characters to integer IDs.

    Standard 20 AAs (alphabetical): A=0, R=1, N=2, D=3, C=4, E=5, Q=6, G=7,
    H=8, I=9, L=10, K=11, M=12, F=13, P=14, S=15, T=16, W=17, Y=18, V=19,
    STOP(*)=20, X=21, PAD=22
    """

    _CHAR_TO_ID = {
        "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "E": 5, "Q": 6, "G": 7,
        "H": 8, "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15,
        "T": 16, "W": 17, "Y": 18, "V": 19, "*": 20, "X": 21,
    }
    _ID_TO_CHAR = {v: k for k, v in _CHAR_TO_ID.items()}
    _ID_TO_CHAR[22] = ""  # PAD decodes to empty
    VOCAB_SIZE = 23
    PAD_ID = 22
    STOP_ID = 20
    X_ID = 21

    def encode(self, sequence: str) -> list[int]:
        """Encode an amino acid string to a list of integer IDs."""
        return [self._CHAR_TO_ID.get(c, 21) for c in sequence.upper()]

    def decode(self, ids: list[int]) -> str:
        """Decode integer IDs back to an amino acid string."""
        return "".join(self._ID_TO_CHAR.get(i, "X") for i in ids)
