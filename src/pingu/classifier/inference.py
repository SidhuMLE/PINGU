"""Production inference wrapper for the AMC classifier.

Loads a trained checkpoint and provides a simple ``classify()`` API that
accepts raw NumPy IQ arrays and returns a predicted modulation type with its
confidence score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pingu.classifier.models.amc_cnn import AMCCNN
from pingu.types import ModulationType


def _select_device() -> torch.device:
    """Select the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AMCInference:
    """Inference wrapper for the AMC CNN classifier.

    Parameters
    ----------
    checkpoint_path : str | Path | None
        Path to a ``.ckpt`` (Lightning) or ``.pt`` (state-dict) file. If
        ``None``, an untrained model is instantiated (useful for testing).
    input_length : int
        Expected number of IQ samples per segment.
    class_names : list[str] | None
        Ordered list of class names corresponding to the model outputs.
    device : torch.device | str | None
        Device override.  If ``None``, automatically selects the best device.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        input_length: int = 1024,
        class_names: list[str] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.input_length = input_length
        self.class_names = class_names or list(AMCCNN.DEFAULT_CLASSES)
        self.num_classes = len(self.class_names)
        self.device = torch.device(device) if device is not None else _select_device()

        # Build the model
        self.model = AMCCNN(input_length=input_length, num_classes=self.num_classes)

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, iq_samples: np.ndarray) -> tuple[ModulationType, float]:
        """Classify a segment of IQ samples.

        Parameters
        ----------
        iq_samples : np.ndarray
            Complex IQ array of shape ``(n_samples,)`` or a 2-row real array
            of shape ``(2, n_samples)`` with I and Q channels.

        Returns
        -------
        tuple[ModulationType, float]
            Predicted modulation type and softmax confidence in ``[0, 1]``.
        """
        tensor = self._prepare_input(iq_samples)

        with torch.no_grad():
            logits = self.model(tensor)                  # (1, num_classes)
            probs = torch.softmax(logits, dim=1)         # (1, num_classes)
            confidence, pred_idx = torch.max(probs, dim=1)

        class_name = self.class_names[pred_idx.item()]
        mod_type = ModulationType(class_name)
        return mod_type, float(confidence.item())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_input(self, iq_samples: np.ndarray) -> torch.Tensor:
        """Convert raw IQ array to a batched ``(1, 2, input_length)`` tensor."""
        if np.iscomplexobj(iq_samples):
            # Complex -> (2, N)
            iq_2ch = np.stack([iq_samples.real, iq_samples.imag], axis=0)
        elif iq_samples.ndim == 2 and iq_samples.shape[0] == 2:
            iq_2ch = iq_samples
        else:
            raise ValueError(
                f"Expected complex 1-D array or (2, N) real array, got shape {iq_samples.shape}"
            )

        # Truncate or zero-pad to input_length
        n = iq_2ch.shape[1]
        if n > self.input_length:
            iq_2ch = iq_2ch[:, : self.input_length]
        elif n < self.input_length:
            pad = np.zeros((2, self.input_length - n), dtype=iq_2ch.dtype)
            iq_2ch = np.concatenate([iq_2ch, pad], axis=1)

        # Normalise to unit power
        power = np.mean(iq_2ch[0] ** 2 + iq_2ch[1] ** 2)
        if power > 0:
            iq_2ch = iq_2ch / np.sqrt(power)

        tensor = torch.from_numpy(iq_2ch.astype(np.float32)).unsqueeze(0)  # (1, 2, L)
        return tensor.to(self.device)

    def _load_checkpoint(self, path: Path) -> None:
        """Load model weights from a Lightning checkpoint or raw state dict."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # Lightning checkpoint: keys are prefixed with "model."
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                # Strip the "model." prefix added by LightningModule
                new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
                state_dict[new_key] = v
            self.model.load_state_dict(state_dict)
        elif isinstance(checkpoint, dict):
            # Raw state dict
            self.model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unrecognised checkpoint format from {path}")
