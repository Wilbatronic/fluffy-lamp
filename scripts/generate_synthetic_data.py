#!/usr/bin/env python3
"""CLI script for generating synthetic training data."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Standard genetic code for translation
GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

STOP_CODONS = {"TAA", "TAG", "TGA"}

# Sample annotations for random assignment
EC_NUMBERS = [
    "1.1.1.1", "1.2.3.4", "2.1.1.2", "2.7.1.1", "3.1.1.1",
    "3.4.11.1", "4.1.1.1", "4.2.1.2", "5.1.1.1", "6.1.1.1",
    "1.14.14.1", "2.3.1.1", "3.5.1.1", "4.3.1.1", "5.4.2.2",
]

KEGG_KOS = [
    "K00001", "K00002", "K00844", "K00845", "K01610",
    "K01689", "K01803", "K01905", "K02358", "K03046",
]

COG_CATEGORIES = ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]


def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(complement.get(base, "N") for base in reversed(seq.upper()))


def generate_coding_sequence(min_length: int, max_length: int, rng: random.Random) -> str:
    """Generate a random coding sequence with proper start and no internal stops."""
    # Target length
    target_nt = rng.randint(min_length, max_length)
    # Round down to multiple of 3 for coding sequence
    num_codons = target_nt // 3
    actual_nt = num_codons * 3

    # Start with ATG
    codons = ["ATG"]

    # Fill with random codons (avoiding stops)
    valid_codons = [c for c in GENETIC_CODE.keys() if c not in STOP_CODONS]
    for _ in range(num_codons - 2):  # -2 for start and stop
        codons.append(rng.choice(valid_codons))

    # End with stop codon
    codons.append(rng.choice(list(STOP_CODONS)))

    return "".join(codons)


def generate_noncoding_sequence(length: int, rng: random.Random) -> str:
    """Generate random non-coding DNA sequence."""
    return "".join(rng.choice("ACGT") for _ in range(length))


def assign_random_annotations(rng: random.Random) -> dict:
    """Assign random functional annotations."""
    # EC number
    ec = rng.choice(EC_NUMBERS)

    # KEGG KO (1-3 KOs)
    num_kos = rng.randint(1, 3)
    kos = ",".join(rng.sample(KEGG_KOS, num_kos))

    # COG categories (1-2 categories)
    num_cogs = rng.randint(1, 2)
    cogs = ",".join(sorted(rng.sample(COG_CATEGORIES, num_cogs)))

    return {
        "ec_number": ec,
        "kegg_ko": kos,
        "cog_category": cogs,
    }


def generate_dataset(
    n_sequences: int,
    min_length: int,
    max_length: int,
    noncoding_fraction: float,
    seed: int,
) -> list[dict]:
    """Generate synthetic dataset."""
    rng = random.Random(seed)

    sequences = []

    for i in range(n_sequences):
        is_coding = rng.random() >= noncoding_fraction

        if is_coding:
            seq = generate_coding_sequence(min_length, max_length, rng)
            # Determine reading frame (0, 1, 2 for forward; 3, 4, 5 for reverse)
            is_rc = rng.random() < 0.5
            frame_offset = rng.randint(0, 2)

            if is_rc:
                seq = reverse_complement(seq)
                reading_frame = 3 + frame_offset
            else:
                reading_frame = frame_offset

            # Shift frame offset for coding sequence
            if frame_offset > 0:
                # Add random prefix to shift frame
                prefix = generate_noncoding_sequence(frame_offset, rng)
                seq = prefix + seq

            # Trim or extend to target length
            target_len = rng.randint(min_length, max_length)
            if len(seq) > target_len:
                seq = seq[:target_len]
            elif len(seq) < target_len:
                seq = seq + generate_noncoding_sequence(target_len - len(seq), rng)

            annotations = assign_random_annotations(rng)

            sequences.append({
                "read_id": f"read_{i:06d}",
                "sequence": seq,
                "is_coding": 1,
                "reading_frame": reading_frame,
                **annotations,
            })
        else:
            # Non-coding sequence
            length = rng.randint(min_length, max_length)
            seq = generate_noncoding_sequence(length, rng)

            sequences.append({
                "read_id": f"read_{i:06d}",
                "sequence": seq,
                "is_coding": 0,
                "reading_frame": -1,
                "ec_number": "",
                "kegg_ko": "",
                "cog_category": "",
            })

    return sequences


def split_dataset(
    sequences: list[dict], train_frac: float = 0.8, val_frac: float = 0.1
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split dataset into train/val/test."""
    n = len(sequences)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = sequences[:n_train]
    val = sequences[n_train : n_train + n_val]
    test = sequences[n_train + n_val :]

    return train, val, test


def write_fasta(sequences: list[dict], output_path: Path) -> None:
    """Write sequences to FASTA file."""
    with open(output_path, "w") as f:
        for seq in sequences:
            f.write(f">{seq['read_id']}\n")
            f.write(f"{seq['sequence']}\n")


def write_labels_tsv(sequences: list[dict], output_path: Path) -> None:
    """Write labels to TSV file."""
    fieldnames = [
        "read_id", "is_coding", "reading_frame",
        "ec_number", "kegg_ko", "cog_category"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for seq in sequences:
            writer.writerow(seq)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for TriFrame."
    )

    parser.add_argument(
        "--output-dir", required=True, help="Output directory for generated data"
    )
    parser.add_argument(
        "--n-sequences", type=int, default=10000, help="Total number of sequences"
    )
    parser.add_argument(
        "--min-length", type=int, default=150, help="Minimum sequence length"
    )
    parser.add_argument(
        "--max-length", type=int, default=300, help="Maximum sequence length"
    )
    parser.add_argument(
        "--noncoding-fraction", type=float, default=0.2,
        help="Fraction of non-coding sequences"
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.8, help="Fraction for training"
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.1, help="Fraction for validation"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_sequences} synthetic sequences...")
    print(f"  Length range: {args.min_length}-{args.max_length}")
    print(f"  Non-coding fraction: {args.noncoding_fraction}")
    print(f"  Random seed: {args.seed}")

    # Generate sequences
    sequences = generate_dataset(
        n_sequences=args.n_sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        noncoding_fraction=args.noncoding_fraction,
        seed=args.seed,
    )

    # Split
    train, val, test = split_dataset(sequences, args.train_frac, args.val_frac)

    print(f"\nDataset split:")
    print(f"  Train: {len(train)} ({100*len(train)/len(sequences):.1f}%)")
    print(f"  Val: {len(val)} ({100*len(val)/len(sequences):.1f}%)")
    print(f"  Test: {len(test)} ({100*len(test)/len(sequences):.1f}%)")

    # Write files
    write_fasta(train, output_dir / "train.fasta")
    write_labels_tsv(train, output_dir / "train_labels.tsv")

    write_fasta(val, output_dir / "val.fasta")
    write_labels_tsv(val, output_dir / "val_labels.tsv")

    write_fasta(test, output_dir / "test.fasta")
    write_labels_tsv(test, output_dir / "test_labels.tsv")

    print(f"\nFiles written to {output_dir}:")
    print(f"  - train.fasta, train_labels.tsv")
    print(f"  - val.fasta, val_labels.tsv")
    print(f"  - test.fasta, test_labels.tsv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
