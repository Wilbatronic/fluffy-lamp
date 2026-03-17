"""Tests for individual model components."""

import pytest
import torch

from triframe.model.config import TriFrameConfig
from triframe.model.nucleotide_encoder import NucleotideEncoder
from triframe.model.frame_encoder import FrameEncoder
from triframe.model.frame_attention import FrameAttention
from triframe.model.cross_resolution import CrossResolutionFusion


class TestNucleotideEncoder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.enc = NucleotideEncoder(d_nucleotide=64, n_layers=4, max_read_length=512)

    def test_output_shape(self):
        x = torch.randint(0, 5, (2, 100))
        out = self.enc(x)
        assert out.shape == (2, 100, 64)

    def test_single_sequence(self):
        x = torch.randint(0, 5, (1, 50))
        out = self.enc(x)
        assert out.shape == (1, 50, 64)

    def test_different_lengths(self):
        for length in [10, 50, 200]:
            x = torch.randint(0, 5, (1, length))
            out = self.enc(x)
            assert out.shape == (1, length, 64)


class TestFrameEncoder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.enc = FrameEncoder(d_frame=64, n_layers=2, n_heads=4, max_aa_length=128)

    def test_output_shape(self):
        aa_ids = torch.randint(0, 22, (2, 6, 50))
        frame_lengths = torch.full((2, 6), 50)
        out = self.enc(aa_ids, frame_lengths)
        assert out.shape == (2, 6, 50, 64)

    def test_shared_weights(self):
        aa_ids = torch.randint(0, 22, (1, 6, 30))
        aa_ids[:, 1] = aa_ids[:, 0]  # frames 0 and 1 identical
        frame_lengths = torch.full((1, 6), 30)

        self.enc.eval()
        with torch.no_grad():
            out = self.enc(aa_ids, frame_lengths)

        torch.testing.assert_close(out[0, 0], out[0, 1], atol=1e-5, rtol=1e-5)


class TestFrameAttention:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.attn = FrameAttention(d_frame=64, n_layers=2, n_heads=4)

    def test_output_shape(self):
        features = torch.randn(2, 6, 50, 64)
        lengths = torch.full((2, 6), 50)
        out, gates = self.attn(features, lengths)
        assert out.shape == (2, 64)
        assert gates.shape == (2, 6)

    def test_gates_sum_to_one(self):
        features = torch.randn(3, 6, 30, 64)
        lengths = torch.full((3, 6), 30)
        _, gates = self.attn(features, lengths)
        sums = gates.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_gates_non_negative(self):
        features = torch.randn(2, 6, 40, 64)
        lengths = torch.full((2, 6), 40)
        _, gates = self.attn(features, lengths)
        assert (gates >= 0).all()


class TestCrossResolutionFusion:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.fusion = CrossResolutionFusion(
            d_nucleotide=64, d_frame=128, n_layers=2, n_heads=4
        )

    def test_output_shape(self):
        nuc = torch.randn(2, 100, 64)
        frame_repr = torch.randn(2, 128)
        mask = torch.zeros(2, 100, dtype=torch.bool)
        out = self.fusion(nuc, frame_repr, mask)
        assert out.shape == (2, 128)

    def test_with_padding(self):
        nuc = torch.randn(2, 100, 64)
        frame_repr = torch.randn(2, 128)
        mask = torch.zeros(2, 100, dtype=torch.bool)
        mask[1, 60:] = True
        out = self.fusion(nuc, frame_repr, mask)
        assert out.shape == (2, 128)

    def test_gradients_flow(self):
        nuc = torch.randn(2, 50, 64, requires_grad=True)
        frame_repr = torch.randn(2, 128, requires_grad=True)
        mask = torch.zeros(2, 50, dtype=torch.bool)
        out = self.fusion(nuc, frame_repr, mask)
        out.sum().backward()
        assert nuc.grad is not None
        assert frame_repr.grad is not None
