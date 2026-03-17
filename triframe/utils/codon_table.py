"""Standard genetic code lookup table and tensor builder."""

import torch

STANDARD_CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_NUC_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}

_AA_CHAR_TO_ID = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "E": 5, "Q": 6, "G": 7,
    "H": 8, "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15,
    "T": 16, "W": 17, "Y": 18, "V": 19, "*": 20,
}

X_ID = 21
PAD_ID = 22


def build_codon_lookup_tensor() -> torch.Tensor:
    """Build a (5, 5, 5) integer tensor mapping nucleotide ID triplets to amino acid IDs.

    codon_table[nuc1][nuc2][nuc3] = aa_id

    Any codon containing N (index 4) maps to X (21).
    Stop codons (TAA, TAG, TGA) map to STOP (20).
    Standard amino acids map to their alphabetical index 0-19.
    """
    table = torch.full((5, 5, 5), X_ID, dtype=torch.long)

    for codon_str, aa_char in STANDARD_CODON_TABLE.items():
        n1, n2, n3 = [_NUC_TO_ID[c] for c in codon_str]
        table[n1, n2, n3] = _AA_CHAR_TO_ID[aa_char]

    return table
