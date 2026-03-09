"""TDoA estimation sub-package.

Public API:
    - GCC methods: gcc_phat, gcc_scot, gcc_ml, estimate_tdoa
    - Peak interpolation: parabolic_interpolation, sinc_interpolation
    - Pair management: PairManager
    - Uncertainty: crlb_tdoa
"""

from pingu.tdoa.gcc import estimate_tdoa, gcc_basic, gcc_ml, gcc_phat, gcc_scot, select_gcc_method
from pingu.tdoa.pair_manager import PairManager
from pingu.tdoa.peak_interpolation import parabolic_interpolation, sinc_interpolation
from pingu.tdoa.uncertainty import crlb_tdoa

__all__ = [
    "gcc_basic",
    "gcc_phat",
    "gcc_scot",
    "gcc_ml",
    "estimate_tdoa",
    "select_gcc_method",
    "parabolic_interpolation",
    "sinc_interpolation",
    "PairManager",
    "crlb_tdoa",
]
