"""PyTorch Lightning module for AMC CNN training and evaluation.

Wraps :class:`~pingu.classifier.models.amc_cnn.AMCCNN` with training /
validation steps, cross-entropy loss, accuracy metrics, and a cosine-annealing
learning-rate schedule.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy

from pingu.classifier.models.amc_cnn import AMCCNN


class AMCLightningModule(pl.LightningModule):
    """Lightning module for automatic modulation classification.

    Parameters
    ----------
    input_length : int
        Number of IQ samples per segment.
    num_classes : int
        Number of modulation classes.
    learning_rate : float
        Initial learning rate for Adam.
    weight_decay : float
        L2 regularisation coefficient.
    t_max : int
        ``T_max`` parameter for ``CosineAnnealingLR`` (typically total epochs).
    class_names : list[str] | None
        Human-readable class names for per-class logging.
    """

    def __init__(
        self,
        input_length: int = 1024,
        num_classes: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        t_max: int = 100,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = AMCCNN(input_length=input_length, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.num_classes = num_classes
        self.class_names = class_names or AMCCNN.DEFAULT_CLASSES[:num_classes]

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc_per_class = Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="none",
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass through the underlying CNN."""
        return self.model(x)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Compute training loss and accuracy for one batch."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Compute validation loss and accuracy for one batch."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc(preds, y)
        self.val_acc_per_class(preds, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Log per-class validation accuracy at the end of each epoch."""
        per_class = self.val_acc_per_class.compute()
        for i, name in enumerate(self.class_names):
            self.log(f"val/acc_{name}", per_class[i], prog_bar=False)
        self.val_acc_per_class.reset()

    # ------------------------------------------------------------------
    # Optimiser and scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict:
        """Configure Adam optimiser with cosine-annealing LR schedule."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.t_max,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
