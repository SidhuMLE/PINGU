#!/usr/bin/env python3
"""Training script for the AMC CNN classifier.

Loads configuration from ``configs/training/amc_baseline.yaml`` (merged on top
of ``configs/default.yaml``), creates the synthetic dataset, and trains the
model using PyTorch Lightning.  The best checkpoint is saved under
``data/models/``.

Usage
-----
    python scripts/train_amc.py
    python scripts/train_amc.py --config configs/training/amc_baseline.yaml
    python scripts/train_amc.py --epochs 20 --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

# Ensure the project root is on sys.path when running as a script
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from pingu.classifier.dataset import AMCDataset
from pingu.classifier.lightning_module import AMCLightningModule
from pingu.config import load_and_merge


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train AMC CNN classifier")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training YAML config (default: configs/training/amc_baseline.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Entry point for AMC training."""
    args = parse_args()

    # ----------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------
    default_cfg_path = _project_root / "configs" / "default.yaml"
    overlay_cfg_path = (
        Path(args.config) if args.config else _project_root / "configs" / "training" / "amc_baseline.yaml"
    )

    cfg = load_and_merge(default_cfg_path, overlay_cfg_path)

    # Apply CLI overrides
    epochs = args.epochs or cfg.training.epochs
    batch_size = args.batch_size or cfg.training.batch_size
    lr = args.lr or cfg.training.learning_rate
    weight_decay = cfg.training.weight_decay
    snr_range = tuple(cfg.training.snr_range_db)
    samples_per_class = cfg.training.samples_per_class
    val_split = cfg.training.val_split
    num_workers = cfg.training.get("num_workers", 0)
    input_length = cfg.classifier.input_length
    class_names = list(cfg.classifier.classes)
    num_classes = len(class_names)

    # Seed everything
    pl.seed_everything(args.seed, workers=True)

    # ----------------------------------------------------------------
    # Dataset and dataloaders
    # ----------------------------------------------------------------
    full_dataset = AMCDataset(
        samples_per_class=samples_per_class,
        snr_range_db=snr_range,
        input_length=input_length,
        modulations=class_names,
        seed=args.seed,
    )

    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    model = AMCLightningModule(
        input_length=input_length,
        num_classes=num_classes,
        learning_rate=lr,
        weight_decay=weight_decay,
        t_max=epochs,
        class_names=class_names,
    )

    # ----------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------
    models_dir = _project_root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(models_dir),
        filename="amc_cnn_best",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )

    # ----------------------------------------------------------------
    # Trainer
    # ----------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        deterministic=True,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val/acc:    {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
