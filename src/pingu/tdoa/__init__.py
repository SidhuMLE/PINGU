"""TDoA estimation sub-package.

Public API:
    - GCC methods: gcc_phat, gcc_scot, gcc_ml, estimate_tdoa
    - Peak interpolation: parabolic_interpolation, sinc_interpolation
    - Pair management: PairManager
    - Uncertainty: crlb_tdoa
"""

from pingu.tdoa.gcc import estimate_tdoa, gcc_ml, gcc_phat, gcc_scot
from pingu.tdoa.pair_manager import PairManager
from pingu.tdoa.peak_interpolation import parabolic_interpolation, sinc_interpolation
from pingu.tdoa.uncertainty import crlb_tdoa

__all__ = [
    "gcc_phat",
    "gcc_scot",
    "gcc_ml",
    "estimate_tdoa",
    "parabolic_interpolation",
    "sinc_interpolation",
    "PairManager",
    "crlb_tdoa",
]
