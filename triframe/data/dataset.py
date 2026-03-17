"""Dataset classes for TriFrame: FASTA reads and synthetic training data."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from triframe.data.tokenizer import DNATokenizer


class FASTAReadDataset(Dataset):
    """Loads DNA reads from a FASTA file.

    Each FASTA record is treated as one read.  An optional TSV label file
    provides functional annotations.

    Args:
        fasta_path: Path to the FASTA file of DNA reads.
        label_path: Optional TSV with columns:
            ``read_id, is_coding, reading_frame, ec_number, kegg_ko, cog_category``.
        max_read_length: Sequences longer than this are truncated.

    Returns per ``__getitem__``:
        dict with ``nucleotide_ids`` (LongTensor), ``length`` (int), and
        optionally ``labels`` (dict).
    """

    def __init__(
        self,
        fasta_path: str | Path,
        label_path: str | Path | None = None,
        max_read_length: int = 2048,
    ):
        self.tokenizer = DNATokenizer()
        self.max_read_length = max_read_length

        self.records: list[tuple[str, str]] = []
        self._load_fasta(fasta_path)

        self.labels: dict[str, dict] | None = None
        if label_path is not None:
            self._load_labels(label_path)

    def _load_fasta(self, path: str | Path) -> None:
        header: str | None = None
        seq_parts: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        self.records.append((header, "".join(seq_parts)))
                    header = line[1:].split()[0]
                    seq_parts = []
                else:
                    seq_parts.append(line)
            if header is not None:
                self.records.append((header, "".join(seq_parts)))

    def _load_labels(self, path: str | Path) -> None:
        self.labels = {}
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.labels[row["read_id"]] = {
                    "is_coding": int(row.get("is_coding", 0)),
                    "reading_frame": int(row.get("reading_frame", -1)),
                    "ec_number": row.get("ec_number", ""),
                    "kegg_ko": row.get("kegg_ko", ""),
                    "cog_category": row.get("cog_category", ""),
                }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        read_id, seq = self.records[idx]
        seq = seq[: self.max_read_length]
        ids = self.tokenizer.encode(seq)
        out: dict = {
            "nucleotide_ids": torch.tensor(ids, dtype=torch.long),
            "length": len(ids),
        }
        if self.labels is not None and read_id in self.labels:
            out["labels"] = self.labels[read_id]
        return out


class SyntheticReadDataset(Dataset):
    """Generates synthetic training data by sampling sub-reads from coding sequences.

    For each sample the dataset:
        1. Picks a random CDS.
        2. Samples a random subsequence of length ``uniform(min_read_len, max_read_len)``.
        3. Optionally reverse-complements (50 % chance).
        4. The correct reading frame is deterministic from the sampling offset.
        5. Labels come from the source CDS annotation.

    A configurable fraction of samples are random non-coding DNA.

    Args:
        fasta_path: Path to a FASTA file of full coding DNA sequences.
        annotations_path: TSV with columns ``seq_id, ec_number, kegg_ko, cog_category``.
        n_samples: Virtual dataset size (samples are generated on the fly).
        min_read_length: Minimum sampled read length.
        max_read_length: Maximum sampled read length.
        include_noncoding: Fraction of non-coding samples.
        seed: Random seed for reproducibility (optional).
    """

    _RC_MAP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

    def __init__(
        self,
        fasta_path: str | Path,
        annotations_path: str | Path,
        n_samples: int = 100_000,
        min_read_length: int = 150,
        max_read_length: int = 300,
        include_noncoding: float = 0.2,
        seed: int | None = None,
    ):
        self.tokenizer = DNATokenizer()
        self.n_samples = n_samples
        self.min_read_length = min_read_length
        self.max_read_length = max_read_length
        self.include_noncoding = include_noncoding
        self.rng = random.Random(seed)

        self.sequences: list[tuple[str, str]] = []
        self._load_fasta(fasta_path)

        self.annotations: dict[str, dict] = {}
        self._load_annotations(annotations_path)

    def _load_fasta(self, path: str | Path) -> None:
        header: str | None = None
        seq_parts: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        self.sequences.append((header, "".join(seq_parts)))
                    header = line[1:].split()[0]
                    seq_parts = []
                else:
                    seq_parts.append(line)
            if header is not None:
                self.sequences.append((header, "".join(seq_parts)))

    def _load_annotations(self, path: str | Path) -> None:
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.annotations[row["seq_id"]] = {
                    "ec_number": row.get("ec_number", ""),
                    "kegg_ko": row.get("kegg_ko", ""),
                    "cog_category": row.get("cog_category", ""),
                }

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        return seq.translate(SyntheticReadDataset._RC_MAP)[::-1]

    def _generate_noncoding(self, length: int) -> str:
        return "".join(self.rng.choice("ACGT") for _ in range(length))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.rng.random() + idx)
        read_len = rng.randint(self.min_read_length, self.max_read_length)

        if rng.random() < self.include_noncoding:
            seq = self._generate_noncoding(read_len)
            ids = self.tokenizer.encode(seq)
            return {
                "nucleotide_ids": torch.tensor(ids, dtype=torch.long),
                "length": len(ids),
                "labels": {
                    "is_coding": 0,
                    "reading_frame": -1,
                    "ec_number": "",
                    "kegg_ko": "",
                    "cog_category": "",
                },
            }

        seq_id, full_seq = rng.choice(self.sequences)
        full_seq = full_seq.upper()

        if len(full_seq) <= read_len:
            start = 0
            sub = full_seq
        else:
            start = rng.randint(0, len(full_seq) - read_len)
            sub = full_seq[start : start + read_len]

        frame_offset = start % 3

        is_rc = rng.random() < 0.5
        if is_rc:
            sub = self._reverse_complement(sub)
            reading_frame = 3 + frame_offset
        else:
            reading_frame = frame_offset

        ids = self.tokenizer.encode(sub)
        ann = self.annotations.get(seq_id, {"ec_number": "", "kegg_ko": "", "cog_category": ""})

        return {
            "nucleotide_ids": torch.tensor(ids, dtype=torch.long),
            "length": len(ids),
            "labels": {
                "is_coding": 1,
                "reading_frame": reading_frame,
                **ann,
            },
        }
