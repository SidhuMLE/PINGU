"""Physical constants and HF band parameters."""

# Speed of light in vacuum (m/s)
SPEED_OF_LIGHT: float = 299_792_458.0

# HF band limits (Hz)
HF_LOWER: float = 3e6   # 3 MHz
HF_UPPER: float = 30e6  # 30 MHz

# Common HF sub-bands (Hz)
HF_BANDS: dict[str, tuple[float, float]] = {
    "80m": (3.5e6, 4.0e6),
    "60m": (5.0e6, 5.5e6),
    "40m": (7.0e6, 7.3e6),
    "30m": (10.1e6, 10.15e6),
    "20m": (14.0e6, 14.35e6),
    "17m": (18.068e6, 18.168e6),
    "15m": (21.0e6, 21.45e6),
    "12m": (24.89e6, 24.99e6),
    "10m": (28.0e6, 29.7e6),
}

# Default sample rate for narrowband processing (Hz)
DEFAULT_NARROWBAND_FS: float = 48_000.0

# Default wideband sample rate (Hz)
DEFAULT_WIDEBAND_FS: float = 2_000_000.0
