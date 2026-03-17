"""Tests for the full TriFrame model: forward pass, shapes, and parameter counts."""

import pytest
import torch

from triframe.model import TriFrameModel, TriFrameConfig


def _run_forward(config: TriFrameConfig, batch_size: int = 2, seq_len: int = 150):
    model = TriFrameModel(config)
    model.eval()
    nuc_ids = torch.randint(0, 4, (batch_size, seq_len))
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    with torch.no_grad():
        return model, model(nuc_ids, lengths)


class TestSmallForward:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = TriFrameConfig.small()
        self.model, self.preds = _run_forward(self.config)

    def test_coding_logits_shape(self):
        assert self.preds["coding_logits"].shape == (2, 2)

    def test_frame_logits_shape(self):
        assert self.preds["frame_logits"].shape == (2, 7)

    def test_ec_logits_shapes(self):
        assert self.preds["ec_logits"]["level1"].shape == (2, self.config.n_ec_level1)
        assert self.preds["ec_logits"]["level2"].shape == (2, self.config.n_ec_level2)
        assert self.preds["ec_logits"]["level3"].shape == (2, self.config.n_ec_level3)
        assert self.preds["ec_logits"]["level4"].shape == (2, self.config.n_ec_level4)

    def test_kegg_logits_shape(self):
        assert self.preds["kegg_logits"].shape == (2, self.config.n_kegg_orthologs)

    def test_cog_logits_shape(self):
        assert self.preds["cog_logits"].shape == (2, self.config.n_cog_categories)

    def test_frame_gates_shape(self):
        assert self.preds["frame_gates"].shape == (2, 6)

    def test_frame_gates_sum_to_one(self):
        sums = self.preds["frame_gates"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_param_count(self):
        count = self.model.count_parameters()
        assert 0.9 * 11e6 <= count <= 1.1 * 11e6, f"Small: {count:,} params"


class TestBaseForward:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = TriFrameConfig.base()
        self.model, self.preds = _run_forward(self.config)

    def test_output_keys(self):
        expected = {"coding_logits", "frame_logits", "ec_logits",
                    "kegg_logits", "cog_logits", "frame_gates"}
        assert expected == set(self.preds.keys())

    def test_frame_gates_sum_to_one(self):
        sums = self.preds["frame_gates"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_param_count(self):
        count = self.model.count_parameters()
        assert 0.9 * 56e6 <= count <= 1.1 * 56e6, f"Base: {count:,} params"


class TestLargeForward:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = TriFrameConfig.large()
        self.model, self.preds = _run_forward(self.config)

    def test_kegg_logits_shape(self):
        assert self.preds["kegg_logits"].shape == (2, 25000)

    def test_frame_gates_sum_to_one(self):
        sums = self.preds["frame_gates"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_param_count(self):
        count = self.model.count_parameters()
        assert 0.9 * 162e6 <= count <= 1.1 * 162e6, f"Large: {count:,} params"


class TestConfigIO:
    def test_from_yaml(self, tmp_path):
        cfg = TriFrameConfig.small()
        path = tmp_path / "cfg.yaml"
        cfg.to_yaml(path)
        loaded = TriFrameConfig.from_yaml(path)
        assert loaded.d_nucleotide == cfg.d_nucleotide
        assert loaded.d_frame == cfg.d_frame

    def test_variable_length_batch(self):
        config = TriFrameConfig.small()
        model = TriFrameModel(config)
        model.eval()
        nuc_ids = torch.randint(0, 4, (3, 200))
        nuc_ids[1, 150:] = 4
        nuc_ids[2, 100:] = 4
        lengths = torch.tensor([200, 150, 100])
        with torch.no_grad():
            preds = model(nuc_ids, lengths)
        assert preds["coding_logits"].shape == (3, 2)
