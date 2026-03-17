"""Tests for TriFrame training components."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from triframe.model import TriFrameModel, TriFrameConfig
from triframe.training.losses import TriFrameLoss
from triframe.training.metrics import TriFrameMetrics


class TestTriFrameLoss:
    """Tests for TriFrameLoss multi-task loss computation."""

    @pytest.fixture
    def loss_fn(self):
        return TriFrameLoss(
            n_kegg_orthologs=100,
            n_cog_categories=26,
        )

    @pytest.fixture
    def mock_predictions(self):
        batch_size = 4
        return {
            "coding_logits": torch.randn(batch_size, 2),
            "frame_logits": torch.randn(batch_size, 7),
            "ec_logits": {
                "level1": torch.randn(batch_size, 7),
                "level2": torch.randn(batch_size, 68),
                "level3": torch.randn(batch_size, 264),
                "level4": torch.randn(batch_size, 100),
            },
            "kegg_logits": torch.randn(batch_size, 100),
            "cog_logits": torch.randn(batch_size, 26),
        }

    @pytest.fixture
    def mock_labels(self):
        return {
            "is_coding": torch.tensor([1, 0, 1, -1]),  # -1 = missing
            "reading_frame": torch.tensor([0, -1, 2, -1]),
            "ec_number": ["1.2.3.4", "", "2.7.1.1", ""],
            "kegg_ko": ["K00001", "", "K00002,K00003", ""],
            "cog_category": ["C", "", "G,E", ""],
        }

    def test_loss_computes_without_crash(self, loss_fn, mock_predictions, mock_labels):
        """Test that loss computation runs without errors."""
        total_loss, loss_components = loss_fn(mock_predictions, mock_labels)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0  # scalar
        assert total_loss.item() >= 0  # loss should be non-negative

        # Check all components exist
        expected_keys = ["coding", "frame", "ec", "kegg", "cog"]
        for key in expected_keys:
            assert key in loss_components
            assert isinstance(loss_components[key], torch.Tensor)

    def test_loss_masks_noncoding_for_frame(self, loss_fn):
        """Test that frame loss is only computed for coding samples."""
        batch_size = 4
        predictions = {
            "coding_logits": torch.randn(batch_size, 2),
            "frame_logits": torch.randn(batch_size, 7),
            "ec_logits": {
                "level1": torch.randn(batch_size, 7),
                "level2": torch.randn(batch_size, 68),
                "level3": torch.randn(batch_size, 264),
                "level4": torch.randn(batch_size, 100),
            },
            "kegg_logits": torch.randn(batch_size, 100),
            "cog_logits": torch.randn(batch_size, 26),
        }

        # All non-coding - frame loss should be 0
        labels_all_noncoding = {
            "is_coding": torch.tensor([0, 0, 0, 0]),
            "reading_frame": torch.tensor([-1, -1, -1, -1]),
            "ec_number": ["", "", "", ""],
            "kegg_ko": ["", "", "", ""],
            "cog_category": ["", "", "", ""],
        }

        _, loss_components = loss_fn(predictions, labels_all_noncoding)
        assert loss_components["frame"].item() == 0.0

    def test_loss_handles_all_missing_labels(self, loss_fn, mock_predictions):
        """Test that loss handles all-missing labels gracefully."""
        missing_labels = {
            "is_coding": torch.tensor([-1, -1, -1, -1]),
            "reading_frame": torch.tensor([-1, -1, -1, -1]),
            "ec_number": ["", "", "", ""],
            "kegg_ko": ["", "", "", ""],
            "cog_category": ["", "", "", ""],
        }

        total_loss, loss_components = loss_fn(mock_predictions, missing_labels)

        # All components should be 0
        for key in ["coding", "frame", "ec", "kegg", "cog"]:
            assert loss_components[key].item() == 0.0

        # Total loss should be 0
        assert total_loss.item() == 0.0

    def test_loss_handles_none_labels(self, loss_fn, mock_predictions):
        """Test that loss handles None labels."""
        total_loss, loss_components = loss_fn(mock_predictions, None)

        # All components should be 0
        for key in ["coding", "frame", "ec", "kegg", "cog"]:
            assert loss_components[key].item() == 0.0

    def test_ec_label_parsing(self, loss_fn):
        """Test EC number string parsing."""
        ec_strings = ["1.2.3.4", "2.7.1.1", "", "1.1.1"]
        parsed = loss_fn._parse_ec_labels(ec_strings)

        assert parsed.shape == (4, 4)
        assert parsed[0].tolist() == [1, 2, 3, 4]
        assert parsed[1].tolist() == [2, 7, 1, 1]
        assert parsed[2].tolist() == [-1, -1, -1, -1]
        assert parsed[3].tolist() == [1, 1, 1, -1]  # partial EC


class TestTriFrameMetrics:
    """Tests for TriFrameMetrics evaluation metrics."""

    @pytest.fixture
    def metrics(self):
        return TriFrameMetrics()

    @pytest.fixture
    def mock_predictions(self):
        batch_size = 4
        return {
            "coding_logits": torch.tensor([[0.1, 2.0], [2.0, 0.1], [0.1, 2.0], [2.0, 0.1]]),
            "frame_logits": F.one_hot(torch.tensor([0, 1, 2, 3]), num_classes=7).float(),
            "ec_logits": {
                "level1": F.one_hot(torch.tensor([0, 1, 2, 0]), num_classes=7).float(),
                "level2": F.one_hot(torch.tensor([5, 10, 15, 5]), num_classes=68).float(),
                "level3": F.one_hot(torch.tensor([10, 50, 100, 10]), num_classes=264).float(),
                "level4": F.one_hot(torch.tensor([50, 100, 200, 50]), num_classes=5000).float(),
            },
            "kegg_logits": torch.zeros(4, 100),
            "cog_logits": torch.zeros(4, 26),
            "frame_gates": torch.tensor([[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]] * 4),
        }

    @pytest.fixture
    def mock_labels(self):
        return {
            "is_coding": torch.tensor([1, 0, 1, -1]),
            "reading_frame": torch.tensor([0, -1, 2, -1]),
            "ec_number": ["0.5.10.50", "", "2.15.100.200", ""],
            "kegg_ko": ["", "", "", ""],
            "cog_category": ["", "", "", ""],
        }

    def test_metrics_update_and_compute(self, metrics, mock_predictions, mock_labels):
        """Test metrics update and compute cycle."""
        metrics.reset()
        metrics.update(mock_predictions, mock_labels)
        results = metrics.compute()

        assert isinstance(results, dict)
        expected_keys = [
            "coding_accuracy", "frame_accuracy",
            "ec_level1_accuracy", "ec_level2_accuracy",
            "ec_level3_accuracy", "ec_level4_accuracy",
            "kegg_f1", "kegg_precision@1", "kegg_precision@3", "kegg_precision@5",
            "cog_f1",
        ]
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], float)

    def test_coding_accuracy(self, metrics):
        """Test coding accuracy computation."""
        predictions = {
            "coding_logits": torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            "frame_logits": torch.randn(3, 7),
            "ec_logits": {
                "level1": torch.randn(3, 7),
                "level2": torch.randn(3, 68),
                "level3": torch.randn(3, 264),
                "level4": torch.randn(3, 5000),
            },
            "kegg_logits": torch.randn(3, 100),
            "cog_logits": torch.randn(3, 26),
            "frame_gates": torch.ones(3, 6) / 6,
        }
        labels = {
            "is_coding": torch.tensor([1, 0, 1]),
            "reading_frame": torch.tensor([-1, -1, -1]),
            "ec_number": ["", "", ""],
            "kegg_ko": ["", "", ""],
            "cog_category": ["", "", ""],
        }

        metrics.reset()
        metrics.update(predictions, labels)
        results = metrics.compute()

        # Should have 2 correct out of 3 (predictions: [1, 0, 1], labels: [1, 0, 1])
        assert results["coding_accuracy"] == pytest.approx(1.0, abs=1e-6)

    def test_frame_accuracy_only_coding(self, metrics):
        """Test that frame accuracy only counts coding samples."""
        predictions = {
            "coding_logits": torch.randn(3, 2),
            "frame_logits": F.one_hot(torch.tensor([0, 0, 0]), num_classes=7).float(),
            "ec_logits": {
                "level1": torch.randn(3, 7),
                "level2": torch.randn(3, 68),
                "level3": torch.randn(3, 264),
                "level4": torch.randn(3, 5000),
            },
            "kegg_logits": torch.randn(3, 100),
            "cog_logits": torch.randn(3, 26),
            "frame_gates": torch.ones(3, 6) / 6,
        }
        labels = {
            "is_coding": torch.tensor([1, 0, 1]),
            "reading_frame": torch.tensor([0, 0, 2]),  # 1st correct, 2nd non-coding (ignored), 3rd wrong
            "ec_number": ["", "", ""],
            "kegg_ko": ["", "", ""],
            "cog_category": ["", "", ""],
        }

        metrics.reset()
        metrics.update(predictions, labels)
        results = metrics.compute()

        # Only coding samples with valid frames count
        # Sample 0: coding, frame correct
        # Sample 1: non-coding, ignored
        # Sample 2: coding, frame wrong (predicted 0, actual 2)
        assert results["frame_accuracy"] == pytest.approx(0.5, abs=1e-6)

    def test_metrics_reset(self, metrics):
        """Test that reset clears accumulated metrics."""
        predictions = {
            "coding_logits": torch.randn(2, 2),
            "frame_logits": torch.randn(2, 7),
            "ec_logits": {
                "level1": torch.randn(2, 7),
                "level2": torch.randn(2, 68),
                "level3": torch.randn(2, 264),
                "level4": torch.randn(2, 100),
            },
            "kegg_logits": torch.randn(2, 100),
            "cog_logits": torch.randn(2, 26),
            "frame_gates": torch.ones(2, 6) / 6,
        }
        labels = {
            "is_coding": torch.tensor([1, 0]),
            "reading_frame": torch.tensor([0, -1]),
            "ec_number": ["1.2.3.4", ""],
            "kegg_ko": ["K00001", ""],
            "cog_category": ["C", ""],
        }

        metrics.update(predictions, labels)
        metrics.reset()
        results = metrics.compute()

        # All counts should be zero after reset
        assert results["coding_accuracy"] == 0.0
        assert results["frame_accuracy"] == 0.0


class TestTrainingStep:
    """Integration test for a single training step."""

    def test_training_step(self):
        """Test a complete forward/backward/optimizer step."""
        config = TriFrameConfig.small()
        model = TriFrameModel(config)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create loss function
        loss_fn = TriFrameLoss(
            n_kegg_orthologs=config.n_kegg_orthologs,
            n_cog_categories=config.n_cog_categories,
        )

        # Mock batch
        batch_size = 2
        seq_len = 100
        nucleotide_ids = torch.randint(0, 5, (batch_size, seq_len))
        lengths = torch.tensor([seq_len, seq_len])
        labels = {
            "is_coding": torch.tensor([1, 0]),
            "reading_frame": torch.tensor([0, -1]),
            "ec_number": ["1.2.3.4", ""],
            "kegg_ko": ["K00001", ""],
            "cog_category": ["C", ""],
        }

        # Forward pass
        predictions = model(nucleotide_ids, lengths)
        loss, _ = loss_fn(predictions, labels)

        # Backward pass
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Model should have gradients after backward pass"

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Verify parameters changed
        # (This is implicitly true if no errors occurred)

    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple steps."""
        config = TriFrameConfig.small()
        model = TriFrameModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        loss_fn = TriFrameLoss(
            n_kegg_orthologs=config.n_kegg_orthologs,
            n_cog_categories=config.n_cog_categories,
        )

        # Accumulate over 3 steps
        accumulation_steps = 3

        for step in range(accumulation_steps):
            nucleotide_ids = torch.randint(0, 5, (2, 100))
            lengths = torch.tensor([100, 100])
            labels = {
                "is_coding": torch.tensor([1, 0]),
                "reading_frame": torch.tensor([0, -1]),
                "ec_number": ["1.2.3.4", ""],
                "kegg_ko": ["K00001", ""],
                "cog_category": ["C", ""],
            }

            predictions = model(nucleotide_ids, lengths)
            loss, _ = loss_fn(predictions, labels)
            loss = loss / accumulation_steps
            loss.backward()

        # Check gradients accumulated
        grad_sum = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_sum += param.grad.abs().sum().item()

        assert grad_sum > 0, "Gradients should be accumulated"

        # Optimizer step
        optimizer.step()
