"""Tests for the AMC classifier components."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pingu.classifier.dataset import AMCDataset
from pingu.classifier.inference import AMCInference
from pingu.classifier.models.amc_cnn import AMCCNN
from pingu.types import ModulationType


# ------------------------------------------------------------------
# CNN model tests
# ------------------------------------------------------------------


class TestAMCCNN:
    """Tests for the AMC CNN model."""

    def test_forward_output_shape(self) -> None:
        """Forward pass should produce (batch, num_classes) logits."""
        batch_size = 4
        input_length = 1024
        num_classes = 8

        model = AMCCNN(input_length=input_length, num_classes=num_classes)
        x = torch.randn(batch_size, 2, input_length)
        logits = model(x)

        assert logits.shape == (batch_size, num_classes), (
            f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
        )

    def test_forward_different_input_lengths(self) -> None:
        """Model should handle various input lengths via adaptive pooling."""
        for length in [256, 512, 1024, 2048]:
            model = AMCCNN(input_length=length, num_classes=8)
            x = torch.randn(2, 2, length)
            logits = model(x)
            assert logits.shape == (2, 8)

    def test_forward_different_num_classes(self) -> None:
        """Model should produce correct number of output classes."""
        for n_cls in [2, 4, 8, 16]:
            model = AMCCNN(input_length=512, num_classes=n_cls)
            x = torch.randn(1, 2, 512)
            logits = model(x)
            assert logits.shape == (1, n_cls)

    def test_gradients_flow(self) -> None:
        """Gradients should flow through the entire network."""
        model = AMCCNN(input_length=1024, num_classes=8)
        x = torch.randn(2, 2, 1024, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None, "Gradients should flow to input"


# ------------------------------------------------------------------
# Dataset tests
# ------------------------------------------------------------------


class TestAMCDataset:
    """Tests for the AMC synthetic dataset."""

    def test_correct_length(self) -> None:
        """Dataset length should equal samples_per_class * num_classes."""
        samples_per_class = 10
        ds = AMCDataset(samples_per_class=samples_per_class, input_length=256, seed=0)
        assert len(ds) == samples_per_class * ds.num_classes

    def test_item_shapes(self) -> None:
        """Each item should be a (2, input_length) tensor with int label."""
        input_length = 512
        ds = AMCDataset(samples_per_class=5, input_length=input_length, seed=0)
        iq, label = ds[0]
        assert isinstance(iq, torch.Tensor)
        assert iq.shape == (2, input_length), f"Expected (2, {input_length}), got {iq.shape}"
        assert isinstance(label, int)

    def test_valid_labels(self) -> None:
        """All labels should be valid class indices."""
        ds = AMCDataset(samples_per_class=5, input_length=256, seed=0)
        for i in range(len(ds)):
            _, label = ds[i]
            assert 0 <= label < ds.num_classes, f"Label {label} out of range [0, {ds.num_classes})"

    def test_all_classes_represented(self) -> None:
        """Every class should appear at least once."""
        ds = AMCDataset(samples_per_class=3, input_length=256, seed=0)
        labels = {ds[i][1] for i in range(len(ds))}
        assert labels == set(range(ds.num_classes)), (
            f"Expected classes {set(range(ds.num_classes))}, got {labels}"
        )

    def test_reproducibility(self) -> None:
        """Same seed should produce identical samples."""
        ds1 = AMCDataset(samples_per_class=3, input_length=256, seed=42)
        ds2 = AMCDataset(samples_per_class=3, input_length=256, seed=42)
        iq1, l1 = ds1[5]
        iq2, l2 = ds2[5]
        assert l1 == l2
        assert torch.allclose(iq1, iq2)

    def test_unit_power_normalisation(self) -> None:
        """Samples should be approximately unit-power after normalisation."""
        ds = AMCDataset(samples_per_class=5, input_length=1024, seed=0)
        for i in range(min(10, len(ds))):
            iq, _ = ds[i]
            power = torch.mean(iq[0] ** 2 + iq[1] ** 2).item()
            assert 0.5 < power < 2.0, f"Sample {i} power = {power:.3f}, expected ~1.0"


# ------------------------------------------------------------------
# Inference wrapper tests
# ------------------------------------------------------------------


class TestAMCInference:
    """Tests for the inference wrapper."""

    def test_classify_complex_input(self) -> None:
        """classify() should accept a complex numpy array and return valid output."""
        infer = AMCInference(checkpoint_path=None, input_length=1024, device="cpu")
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        mod_type, confidence = infer.classify(iq)

        assert isinstance(mod_type, ModulationType)
        assert 0.0 <= confidence <= 1.0

    def test_classify_two_channel_input(self) -> None:
        """classify() should accept a (2, N) real numpy array."""
        infer = AMCInference(checkpoint_path=None, input_length=512, device="cpu")
        iq_2ch = np.random.randn(2, 512).astype(np.float32)
        mod_type, confidence = infer.classify(iq_2ch)

        assert isinstance(mod_type, ModulationType)
        assert 0.0 <= confidence <= 1.0

    def test_classify_short_input_pads(self) -> None:
        """Short inputs should be zero-padded without error."""
        infer = AMCInference(checkpoint_path=None, input_length=1024, device="cpu")
        iq = (np.random.randn(256) + 1j * np.random.randn(256)).astype(np.complex64)
        mod_type, confidence = infer.classify(iq)
        assert isinstance(mod_type, ModulationType)

    def test_classify_long_input_truncates(self) -> None:
        """Long inputs should be truncated without error."""
        infer = AMCInference(checkpoint_path=None, input_length=512, device="cpu")
        iq = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)
        mod_type, confidence = infer.classify(iq)
        assert isinstance(mod_type, ModulationType)

    def test_invalid_input_raises(self) -> None:
        """Invalid input shapes should raise ValueError."""
        infer = AMCInference(checkpoint_path=None, input_length=1024, device="cpu")
        with pytest.raises(ValueError, match="shape"):
            infer.classify(np.random.randn(3, 512).astype(np.float32))

    def test_missing_checkpoint_raises(self) -> None:
        """Non-existent checkpoint path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AMCInference(checkpoint_path="/nonexistent/model.ckpt")
