"""Tests for six-frame translation correctness."""

import pytest
import torch

from triframe.model.frame_translator import SixFrameTranslator
from triframe.data.tokenizer import DNATokenizer, AATokenizer
from triframe.utils.codon_table import build_codon_lookup_tensor


@pytest.fixture
def translator():
    return SixFrameTranslator(max_aa_length=512)


@pytest.fixture
def dna_tok():
    return DNATokenizer()


@pytest.fixture
def aa_tok():
    return AATokenizer()


class TestCodonTable:
    def test_atg_is_methionine(self):
        table = build_codon_lookup_tensor()
        assert table[0, 3, 2].item() == 12  # A=0,T=3,G=2 -> M=12

    def test_taa_is_stop(self):
        table = build_codon_lookup_tensor()
        assert table[3, 0, 0].item() == 20  # T=3,A=0,A=0 -> STOP=20

    def test_tag_is_stop(self):
        table = build_codon_lookup_tensor()
        assert table[3, 0, 2].item() == 20  # T=3,A=0,G=2 -> STOP=20

    def test_tga_is_stop(self):
        table = build_codon_lookup_tensor()
        assert table[3, 2, 0].item() == 20  # T=3,G=2,A=0 -> STOP=20

    def test_n_maps_to_x(self):
        table = build_codon_lookup_tensor()
        assert table[4, 0, 0].item() == 21  # N=4 -> X=21
        assert table[0, 4, 0].item() == 21
        assert table[0, 0, 4].item() == 21

    def test_table_shape(self):
        table = build_codon_lookup_tensor()
        assert table.shape == (5, 5, 5)


class TestReverseComplement:
    def test_basic_rc(self, translator, dna_tok):
        seq = "ATGC"
        ids = torch.tensor([dna_tok.encode(seq)])
        rc = translator._reverse_complement(ids)
        decoded = dna_tok.decode(rc[0].tolist())
        assert decoded == "GCAT"

    def test_rc_of_rc_is_original(self, translator, dna_tok):
        seq = "ATGCGATCGA"
        ids = torch.tensor([dna_tok.encode(seq)])
        rc = translator._reverse_complement(ids)
        rc_rc = translator._reverse_complement(rc)
        decoded = dna_tok.decode(rc_rc[0].tolist())
        assert decoded == seq

    def test_rc_preserves_n(self, translator, dna_tok):
        seq = "ANGC"
        ids = torch.tensor([dna_tok.encode(seq)])
        rc = translator._reverse_complement(ids)
        decoded = dna_tok.decode(rc[0].tolist())
        assert decoded == "GCNT"


class TestSixFrameTranslation:
    def test_known_sequence_frame0(self, translator, dna_tok, aa_tok):
        seq = "ATGGCTTAA"  # ATG=M, GCT=A, TAA=*
        ids = torch.tensor([dna_tok.encode(seq)])
        aa_ids, frame_lengths = translator(ids)

        frame0_ids = aa_ids[0, 0, : frame_lengths[0, 0]].tolist()
        decoded = aa_tok.decode(frame0_ids)
        assert decoded == "MA*"

    def test_frame1_offset(self, translator, dna_tok, aa_tok):
        seq = "XATGGCTX"  # offset 1: ATG=M, GCT=A  (skip first X, last X unused)
        dna = "AATGGCTA"  # offset 1: ATG=M, GCT=A
        ids = torch.tensor([dna_tok.encode(dna)])
        aa_ids, frame_lengths = translator(ids)

        frame1_ids = aa_ids[0, 1, : frame_lengths[0, 1]].tolist()
        decoded = aa_tok.decode(frame1_ids)
        assert decoded == "MA"

    def test_output_shapes(self, translator, dna_tok):
        seq = "ATGGCTTAAGCGATCG"
        ids = torch.tensor([dna_tok.encode(seq)])
        aa_ids, frame_lengths = translator(ids)

        assert aa_ids.shape == (1, 6, 512)
        assert frame_lengths.shape == (1, 6)

    def test_batch_translation(self, translator):
        ids = torch.randint(0, 4, (4, 150))
        aa_ids, frame_lengths = translator(ids)

        assert aa_ids.shape[0] == 4
        assert aa_ids.shape[1] == 6
        assert frame_lengths.shape == (4, 6)
        assert (frame_lengths > 0).all()

    def test_frame_lengths_correct(self, translator):
        seq_len = 150
        ids = torch.randint(0, 4, (1, seq_len))
        _, frame_lengths = translator(ids)

        for offset in range(3):
            expected = (seq_len - offset) // 3
            expected = min(expected, 512)
            assert frame_lengths[0, offset].item() == expected
            assert frame_lengths[0, offset + 3].item() == expected

    def test_n_nucleotide_handling(self, translator, dna_tok):
        seq = "NNNACGTNN"  # NNN -> X, ACG -> T, TNN -> X
        ids = torch.tensor([dna_tok.encode(seq)])
        aa_ids, frame_lengths = translator(ids)

        frame0 = aa_ids[0, 0, : frame_lengths[0, 0]].tolist()
        assert frame0[0] == 21  # NNN -> X
        assert frame0[2] == 21  # TNN -> X
