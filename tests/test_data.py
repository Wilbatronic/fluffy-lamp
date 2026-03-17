"""Tests for dataset classes, tokenizers, and collation."""

import os
import tempfile

import pytest
import torch

from triframe.data.tokenizer import DNATokenizer, AATokenizer
from triframe.data.dataset import FASTAReadDataset, SyntheticReadDataset
from triframe.data.collator import TriFrameCollator


class TestDNATokenizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tok = DNATokenizer()

    def test_encode_basic(self):
        assert self.tok.encode("ACGTN") == [0, 1, 2, 3, 4]

    def test_case_insensitive(self):
        assert self.tok.encode("acgtn") == [0, 1, 2, 3, 4]

    def test_decode_roundtrip(self):
        seq = "ACGTNNACGT"
        assert self.tok.decode(self.tok.encode(seq)) == seq

    def test_unknown_char_maps_to_n(self):
        assert self.tok.encode("X") == [4]


class TestAATokenizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tok = AATokenizer()

    def test_encode_basic(self):
        ids = self.tok.encode("ARNDCEQGHILKMFPSTWYV")
        assert ids == list(range(20))

    def test_stop_and_x(self):
        assert self.tok.encode("*") == [20]
        assert self.tok.encode("X") == [21]

    def test_decode_roundtrip(self):
        seq = "MAGIC"
        assert self.tok.decode(self.tok.encode(seq)) == seq

    def test_vocab_size(self):
        assert self.tok.VOCAB_SIZE == 23


@pytest.fixture
def fasta_file(tmp_path):
    path = tmp_path / "reads.fasta"
    path.write_text(
        ">read1\nATGGCTTAA\n>read2\nGCGATCGATCGA\n>read3\nTTTTTTTTT\n"
    )
    return str(path)


@pytest.fixture
def label_file(tmp_path):
    path = tmp_path / "labels.tsv"
    path.write_text(
        "read_id\tis_coding\treading_frame\tec_number\tkegg_ko\tcog_category\n"
        "read1\t1\t0\t1.1.1.1\tK00001\tC\n"
        "read2\t1\t2\t2.7.1.1\tK00002\tG\n"
        "read3\t0\t-1\t\t\t\n"
    )
    return str(path)


class TestFASTAReadDataset:
    def test_length(self, fasta_file):
        ds = FASTAReadDataset(fasta_file)
        assert len(ds) == 3

    def test_item_keys_no_labels(self, fasta_file):
        ds = FASTAReadDataset(fasta_file)
        item = ds[0]
        assert "nucleotide_ids" in item
        assert "length" in item
        assert "labels" not in item

    def test_item_with_labels(self, fasta_file, label_file):
        ds = FASTAReadDataset(fasta_file, label_path=label_file)
        item = ds[0]
        assert "labels" in item
        assert item["labels"]["is_coding"] == 1

    def test_sequence_encoding(self, fasta_file):
        ds = FASTAReadDataset(fasta_file)
        item = ds[0]  # "ATGGCTTAA"
        tok = DNATokenizer()
        expected = tok.encode("ATGGCTTAA")
        assert item["nucleotide_ids"].tolist() == expected


@pytest.fixture
def cds_fasta(tmp_path):
    path = tmp_path / "cds.fasta"
    path.write_text(
        ">gene1\n" + "ATGGCTTAA" * 50 + "\n"
        ">gene2\n" + "GCGATCGATCGA" * 30 + "\n"
    )
    return str(path)


@pytest.fixture
def cds_annotations(tmp_path):
    path = tmp_path / "annotations.tsv"
    path.write_text(
        "seq_id\tec_number\tkegg_ko\tcog_category\n"
        "gene1\t1.1.1.1\tK00001\tC\n"
        "gene2\t2.7.1.1\tK00002\tG\n"
    )
    return str(path)


class TestSyntheticReadDataset:
    def test_length(self, cds_fasta, cds_annotations):
        ds = SyntheticReadDataset(
            cds_fasta, cds_annotations, n_samples=100, seed=42
        )
        assert len(ds) == 100

    def test_item_has_labels(self, cds_fasta, cds_annotations):
        ds = SyntheticReadDataset(
            cds_fasta, cds_annotations, n_samples=50, seed=42
        )
        item = ds[0]
        assert "labels" in item
        assert "is_coding" in item["labels"]
        assert "reading_frame" in item["labels"]

    def test_read_length_range(self, cds_fasta, cds_annotations):
        ds = SyntheticReadDataset(
            cds_fasta,
            cds_annotations,
            n_samples=200,
            min_read_length=150,
            max_read_length=300,
            include_noncoding=0.0,
            seed=42,
        )
        for i in range(200):
            item = ds[i]
            length = item["length"]
            assert 150 <= length <= 300

    def test_noncoding_fraction(self, cds_fasta, cds_annotations):
        ds = SyntheticReadDataset(
            cds_fasta,
            cds_annotations,
            n_samples=500,
            include_noncoding=0.5,
            seed=42,
        )
        noncoding = sum(1 for i in range(500) if ds[i]["labels"]["is_coding"] == 0)
        assert 150 < noncoding < 350  # roughly 50 %, loose bounds

    def test_coding_frame_valid(self, cds_fasta, cds_annotations):
        ds = SyntheticReadDataset(
            cds_fasta,
            cds_annotations,
            n_samples=100,
            include_noncoding=0.0,
            seed=42,
        )
        for i in range(100):
            item = ds[i]
            assert item["labels"]["reading_frame"] in range(6)


class TestTriFrameCollator:
    def test_padding(self, fasta_file):
        ds = FASTAReadDataset(fasta_file)
        collator = TriFrameCollator()
        batch = collator([ds[i] for i in range(len(ds))])

        assert batch["nucleotide_ids"].shape[0] == 3
        assert batch["nucleotide_ids"].shape[1] == 12  # max length in batch
        assert batch["lengths"].tolist() == [9, 12, 9]

    def test_pad_value(self, fasta_file):
        ds = FASTAReadDataset(fasta_file)
        collator = TriFrameCollator()
        batch = collator([ds[i] for i in range(len(ds))])
        assert (batch["nucleotide_ids"][0, 9:] == 4).all()

    def test_labels_collated(self, fasta_file, label_file):
        ds = FASTAReadDataset(fasta_file, label_path=label_file)
        collator = TriFrameCollator()
        batch = collator([ds[i] for i in range(len(ds))])

        assert batch["labels"] is not None
        assert batch["labels"]["is_coding"].tolist() == [1, 1, 0]
