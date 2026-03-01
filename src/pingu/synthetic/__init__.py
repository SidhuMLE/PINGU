"""Synthetic data engine for PINGU.

Provides signal generators, noise utilities, TDoA scenario simulation,
and channel model placeholders for training the AMC classifier and
validating the geolocation pipeline.
"""

from pingu.synthetic.signals import (
    generate_signal,
    generate_ssb,
    generate_cw,
    generate_am,
    generate_fsk,
    generate_fsk2,
    generate_fsk4,
    generate_bpsk,
    generate_qpsk,
    GENERATORS,
)
from pingu.synthetic.noise import add_awgn, generate_noise
from pingu.synthetic.scenarios import TDoAScenario
from pingu.synthetic.channels import apply_watterson

__all__ = [
    "generate_signal",
    "generate_ssb",
    "generate_cw",
    "generate_am",
    "generate_fsk",
    "generate_fsk2",
    "generate_fsk4",
    "generate_bpsk",
    "generate_qpsk",
    "GENERATORS",
    "add_awgn",
    "generate_noise",
    "TDoAScenario",
    "apply_watterson",
]
