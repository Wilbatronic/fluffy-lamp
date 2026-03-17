"""Tests for TriFrame inference components."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from triframe.model import TriFrameModel, TriFrameConfig
from triframe.inference import TriFramePredictor


class TestTriFramePredictor:
    """Tests for TriFramePredictor inference API."""

    @pytest.fixture
    def small_model(self):
        config = TriFrameConfig.small()
        return TriFrameModel(config)

    @pytest.fixture
    def predictor(self, small_model):
        return TriFramePredictor(small_model, device="cpu")

    def test_predictor_instantiation(self, small_model):
        """Test predictor can be created from model."""
        predictor = TriFramePredictor(small_model, device="cpu")
        assert predictor.model is small_model
        assert predictor.device.type == "cpu"

    def test_predict_reads_returns_correct_format(self, predictor):
        """Test predict_reads returns correctly formatted output."""
        sequences = ["ATGCGTACGTACGTACGTAC", "CGTACGTACGTACGTACGTA"]
        results = predictor.predict_reads(sequences)

        assert len(results) == 2

        for i, result in enumerate(results):
            # Check all expected keys exist
            expected_keys = [
                "read_id", "is_coding", "coding_confidence",
                "predicted_frame", "frame_confidences",
                "ec_number", "ec_confidence",
                "kegg_orthologs", "kegg_scores",
                "cog_categories", "cog_scores",
            ]
            for key in expected_keys:
                assert key in result, f"Missing key: {key}"

            # Check types
            assert isinstance(result["read_id"], str)
            assert isinstance(result["is_coding"], bool)
            assert isinstance(result["coding_confidence"], float)
            assert isinstance(result["predicted_frame"], int)
            assert isinstance(result["frame_confidences"], list)
            assert len(result["frame_confidences"]) == 6
            assert isinstance(result["ec_number"], (str, type(None)))
            assert isinstance(result["ec_confidence"], float)
            assert isinstance(result["kegg_orthologs"], list)
            assert isinstance(result["kegg_scores"], list)
            assert isinstance(result["cog_categories"], list)
            assert isinstance(result["cog_scores"], list)

    def test_predict_single_read(self, predictor):
        """Test predicting a single read."""
        sequence = "ATG" + "CGT" * 50  # 153 bp, starts with ATG
        results = predictor.predict_reads([sequence])

        assert len(results) == 1
        result = results[0]

        # Check structure
        assert result["is_coding"] in [True, False]
        assert 0 <= result["predicted_frame"] <= 5 or result["predicted_frame"] == -1
        assert 0 <= result["coding_confidence"] <= 1
        assert len(result["frame_confidences"]) == 6

        # Frame confidences should sum to approximately 1
        gate_sum = sum(result["frame_confidences"])
        assert gate_sum == pytest.approx(1.0, abs=0.01)

    def test_predict_batch(self, predictor):
        """Test batch prediction with different sequence lengths."""
        sequences = [
            "ATG" * 30,  # 90 bp
            "ATGC" * 50,  # 200 bp
            "ATGCGT" * 40,  # 240 bp
        ]
        results = predictor.predict_reads(sequences, batch_size=2)

        assert len(results) == 3

    def test_predict_very_short_read(self, predictor):
        """Test prediction on very short read (3bp - just start codon)."""
        sequence = "ATG"  # Minimum coding sequence
        results = predictor.predict_reads([sequence])

        assert len(results) == 1
        result = results[0]

        # Should still return valid structure
        assert "read_id" in result
        assert "is_coding" in result
        assert "predicted_frame" in result

    def test_predict_all_n_read(self, predictor):
        """Test prediction on all-N read (shouldn't crash)."""
        sequence = "N" * 150
        results = predictor.predict_reads([sequence])

        assert len(results) == 1
        result = results[0]

        # Should return valid structure even for all-N
        assert "read_id" in result
        assert isinstance(result["is_coding"], bool)

    def test_predict_invalid_bases(self, predictor):
        """Test prediction with invalid bases (should map to N)."""
        sequence = "XYZ" * 50  # Invalid bases
        results = predictor.predict_reads([sequence])

        assert len(results) == 1

    def test_predict_fasta(self, predictor):
        """Test prediction from FASTA file."""
        # Create temporary FASTA
        fasta_content = """>read_001
ATGCGTACGTACGTACGTACGTACGT
>read_002
CGTACGTACGTACGTACGTACGTACG
>read_003
TACGTACGTACGTACGTACGTACGTA
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(fasta_content)
            fasta_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            output_path = f.name

        try:
            results = predictor.predict_fasta(fasta_path, batch_size=2, output_path=output_path)

            assert len(results) == 3

            # Check read IDs from FASTA
            assert results[0]["read_id"] == "read_001"
            assert results[1]["read_id"] == "read_002"
            assert results[2]["read_id"] == "read_003"

            # Check output file was written
            assert Path(output_path).exists()
            content = Path(output_path).read_text()
            assert "read_001" in content
            assert "is_coding" in content

        finally:
            Path(fasta_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_predictor_from_checkpoint(self, small_model):
        """Test loading predictor from checkpoint."""
        # Create a checkpoint
        checkpoint = {
            "model_state_dict": small_model.state_dict(),
            "model_config": small_model.config,
            "epoch": 0,
            "global_step": 0,
            "best_metric": 0.0,
            "training_config": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = f.name

        try:
            predictor = TriFramePredictor.from_checkpoint(checkpoint_path, device="cpu")

            assert predictor is not None
            assert predictor.model is not None

            # Test inference
            results = predictor.predict_reads(["ATGCGTACGT" * 10])
            assert len(results) == 1

        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


class TestInferenceEdgeCases:
    """Tests for inference edge cases."""

    @pytest.fixture
    def predictor(self):
        config = TriFrameConfig.small()
        model = TriFrameModel(config)
        return TriFramePredictor(model, device="cpu")

    def test_empty_sequences_list(self, predictor):
        """Test prediction on empty list."""
        results = predictor.predict_reads([])
        assert results == []

    def test_single_base(self, predictor):
        """Test on single base pair (extreme edge case)."""
        sequences = ["A"]
        results = predictor.predict_reads(sequences)
        assert len(results) == 1

    def test_long_sequence(self, predictor):
        """Test on longer sequence."""
        sequence = "ATG" + "CGT" * 200  # 603 bp
        results = predictor.predict_reads([sequence])
        assert len(results) == 1

    def test_mixed_case_input(self, predictor):
        """Test that mixed case input is handled."""
        sequences = ["AtGcGtAcGt", "atgcgtacgt", "ATGCGTACGT"]
        results = predictor.predict_reads(sequences)
        assert len(results) == 3

    def test_frame_confidences_sum_to_one(self, predictor):
        """Verify that frame gate values sum to approximately 1."""
        sequence = "ATG" + "CGT" * 50
        results = predictor.predict_reads([sequence])

        gates = results[0]["frame_confidences"]
        assert len(gates) == 6
        assert sum(gates) == pytest.approx(1.0, abs=0.01)

    def test_confidence_values_in_range(self, predictor):
        """Verify confidence values are in valid range [0, 1]."""
        sequences = ["ATGCGTACGT" * 15, "CGTACGTACG" * 15]
        results = predictor.predict_reads(sequences)

        for result in results:
            assert 0 <= result["coding_confidence"] <= 1
            assert 0 <= result["ec_confidence"] <= 1
            for score in result["kegg_scores"]:
                assert 0 <= score <= 1
            for score in result["cog_scores"]:
                assert 0 <= score <= 1


class TestCheckpointCompatibility:
    """Tests for checkpoint loading compatibility."""

    def test_checkpoint_with_model_config(self):
        """Test loading checkpoint with model_config field."""
        config = TriFrameConfig.small()
        model = TriFrameModel(config)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": config,
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = f.name

        try:
            predictor = TriFramePredictor.from_checkpoint(checkpoint_path, device="cpu")
            assert predictor.model.config.d_frame == config.d_frame
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)
