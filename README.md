# PINGU — HF TDoA Geolocation Pipeline

PINGU is a hybrid traditional/ML signal processing pipeline for geolocating stationary HF radio transmitters using **Time Difference of Arrival (TDoA)** measurements across distributed receivers.

The pipeline uses traditional DSP for channelization, cross-correlation, Bayesian filtering, and non-linear position solving, while reserving ML for signal detection and modulation classification where synthetic training data is practical.

## Quick Start

```bash
# Requirements: Python 3.11+, uv
uv sync --extra dev

# Run tests (168 unit tests)
uv run pytest

# Run the pipeline on a synthetic scenario
uv run python scripts/run_pipeline.py --snr 20 --n-frames 50

# Generate synthetic training data
uv run python scripts/generate_synthetic.py --n-scenarios 100 --output data/synthetic/

# Train the modulation classifier
uv run python scripts/train_amc.py --epochs 50
```

## How It Works

The pipeline processes wideband IQ data from 5 GPS-synchronized receivers through six stages:

```
Wideband IQ ──► Channelize ──► Detect ──► TDoA Estimate ──► Kalman Filter ──► Position Solve
  (per rx)       (narrowband)   (energy)   (GCC-PHAT)       (integrate)       (NLLS)
```

1. **Channelize** — Split wideband IQ into narrowband channels (polyphase filterbank or FFT)
2. **Detect** — CFAR energy detector identifies active channels
3. **Classify** — CNN classifies modulation type (SSB, CW, AM, FSK, PSK)
4. **TDoA Estimate** — Generalized cross-correlation (GCC-PHAT/SCOT/ML) between all 10 receiver pairs with sub-sample interpolation
5. **Integrate** — Kalman filter fuses TDoA estimates over time (stationary target assumption reduces variance as 1/N)
6. **Locate** — Weighted non-linear least squares (Levenberg-Marquardt) solves for transmitter position with grid-search initialization

## Project Structure

```
PINGU/
├── pyproject.toml                    # Dependencies and build config
├── configs/
│   ├── default.yaml                  # Master configuration
│   ├── training/amc_baseline.yaml    # AMC CNN training hyperparameters
│   └── geometry/pentagon_5rx.yaml    # 5-receiver layout preset
│
├── src/pingu/
│   ├── types.py              # Shared dataclasses (IQFrame, TDoAEstimate, etc.)
│   ├── constants.py          # Physical constants (c, HF bands)
│   ├── config.py             # OmegaConf YAML config loader
│   │
│   ├── channelizer/          # Stage 1: Wideband → narrowband
│   │   ├── base.py           #   Abstract channelizer interface
│   │   ├── polyphase.py      #   Polyphase filterbank (primary)
│   │   └── fft.py            #   FFT overlap-save (reference)
│   │
│   ├── detector/             # Stage 2a: Signal detection
│   │   └── energy.py         #   CFAR energy detector
│   │
│   ├── classifier/           # Stage 2b: Modulation classification
│   │   ├── models/amc_cnn.py #   Conv1d CNN (2-ch IQ input)
│   │   ├── dataset.py        #   On-the-fly synthetic PyTorch Dataset
│   │   ├── lightning_module.py  # PyTorch Lightning training wrapper
│   │   └── inference.py      #   Production inference (CPU/MPS/CUDA)
│   │
│   ├── tdoa/                 # Stage 3: Time-difference estimation
│   │   ├── gcc.py            #   GCC-PHAT, GCC-SCOT, GCC-ML
│   │   ├── peak_interpolation.py  # Parabolic + sinc sub-sample refinement
│   │   ├── pair_manager.py   #   Manages C(n,2) = 10 receiver pairs
│   │   └── uncertainty.py    #   Cramér-Rao Lower Bound
│   │
│   ├── integrator/           # Stage 4: Bayesian temporal filtering
│   │   ├── kalman.py         #   Kalman filter (identity dynamics)
│   │   └── convergence.py    #   Variance tracking + CRLB comparison
│   │
│   ├── locator/              # Stage 5: Position estimation
│   │   ├── geometry.py       #   Receiver positions + TDoA geometry
│   │   ├── cost_functions.py #   Weighted NLLS residuals + Jacobian
│   │   ├── solvers.py        #   Levenberg-Marquardt solver
│   │   └── posterior.py      #   Confidence ellipses
│   │
│   ├── synthetic/            # Data generation
│   │   ├── signals.py        #   SSB, CW, AM, FSK2/4, BPSK, QPSK
│   │   ├── noise.py          #   Calibrated AWGN
│   │   ├── scenarios.py      #   Multi-receiver TDoA scenarios
│   │   └── channels.py       #   Watterson channel model (placeholder)
│   │
│   ├── pipeline/
│   │   └── runner.py         #   End-to-end orchestrator
│   │
│   └── visualization/
│       ├── spectrogram.py    #   IQ spectrogram plots
│       ├── correlation.py    #   Cross-correlation plots
│       ├── map_plot.py       #   Position map + confidence ellipses
│       └── convergence.py    #   Variance-over-time plots
│
├── tests/                    # 168 unit tests (mirrors src/ structure)
├── scripts/
│   ├── run_pipeline.py       #   Run pipeline on synthetic scenario
│   ├── train_amc.py          #   Train modulation classifier
│   └── generate_synthetic.py #   Generate synthetic datasets
└── data/                     # .gitignore'd runtime data
```

## Module Details

### Shared Types (`pingu.types`)

All pipeline stages communicate through typed dataclasses:

| Type | Description |
|------|-------------|
| `IQFrame` | Block of complex IQ samples from one receiver (complex64) |
| `ChannelizedFrame` | Narrowband channels from the channelizer |
| `Detection` | A detected signal with channel, SNR, and optional modulation |
| `TDoAEstimate` | Time delay between a receiver pair with variance |
| `IntegratedTDoA` | Kalman-filtered TDoA vector with covariance matrix |
| `PositionEstimate` | Solved position with 2x2 covariance and confidence radius |
| `ReceiverConfig` | Receiver location (lat/lon or Cartesian) and parameters |
| `ModulationType` | Enum: SSB, CW, AM, FSK2, FSK4, BPSK, QPSK, NOISE |

### Channelizer (`pingu.channelizer`)

Splits wideband IQ into narrowband channels for per-signal processing.

- **PolyphaseChannelizer** — M-path polyphase filterbank with configurable overlap factor and prototype filter. Near-perfect reconstruction. This is the primary channelizer.
- **FFTChannelizer** — Simpler overlap-save implementation for reference and validation.

```python
from pingu.channelizer.polyphase import PolyphaseChannelizer

ch = PolyphaseChannelizer(n_channels=64, overlap_factor=4)
result = ch.channelize(iq_frame)  # → ChannelizedFrame
# result.channels: shape (64, n_samples_per_channel)
```

### Detector (`pingu.detector`)

CFAR (Constant False Alarm Rate) energy detector scans channelized frames for active signals.

```python
from pingu.detector.energy import EnergyDetector

det = EnergyDetector(pfa=1e-6, block_size=1024, guard_cells=4, reference_cells=16)
detections = det.detect(channelized_frame)  # → list[Detection]
```

### Classifier (`pingu.classifier`)

CNN-based Automatic Modulation Classification (AMC) distinguishes between 8 signal types.

- **Architecture**: 3 Conv1d layers (64→128→256), BatchNorm, ReLU, global average pooling, FC head
- **Input**: 2-channel (I, Q) 1D time series, shape `(batch, 2, 1024)`
- **Training**: On-the-fly synthetic data at SNR range [-10, +30] dB via PyTorch Lightning

```python
from pingu.classifier.inference import AMCInference

clf = AMCInference(checkpoint_path="data/models/best.ckpt")
mod_type, confidence = clf.classify(iq_samples)  # → (ModulationType.BPSK, 0.97)
```

### TDoA Estimation (`pingu.tdoa`)

Estimates time differences of arrival between all receiver pairs using generalized cross-correlation.

- **GCC-PHAT** — Phase transform weighting (best for broadband signals)
- **GCC-SCOT** — Smoothed coherence transform
- **GCC-ML** — Maximum-likelihood weighting
- **Sub-sample interpolation** — Parabolic (fast) or sinc (accurate) refinement
- **CRLB** — Cramér-Rao Lower Bound: `Var(Δτ) ≥ 6 / (8π²B³·T·SNR)` for flat-spectrum signals

```python
from pingu.tdoa.gcc import estimate_tdoa
from pingu.tdoa.pair_manager import PairManager

pm = PairManager(["RX0", "RX1", "RX2", "RX3", "RX4"])  # 10 pairs
tdoas = pm.estimate_all_tdoas(signals, fs=48000, method="phat")
```

### Bayesian Integration (`pingu.integrator`)

Kalman filter fuses TDoA measurements over time for a stationary target.

- **State**: 10-vector of TDoA values (one per pair)
- **Dynamics**: Identity matrix (target doesn't move)
- **Joseph form** covariance update for numerical stability
- Variance decreases as ~1/N with N measurement updates

```python
from pingu.integrator.kalman import TDoAKalmanFilter
from pingu.integrator.convergence import ConvergenceMonitor

kf = TDoAKalmanFilter(n_pairs=10, process_noise=1e-12)
monitor = ConvergenceMonitor(n_pairs=10)

for measurement, noise_var in observations:
    kf.predict()
    kf.update(measurement, noise_var)
    monitor.update(kf.get_state().covariance)
    if monitor.is_converged():
        break
```

### Position Estimation (`pingu.locator`)

Solves for transmitter position from integrated TDoA measurements.

- **Cost function**: Weighted NLLS — `p* = argmin Σ (1/σ²_ij) · ||Δτ̂_ij - (1/c)(||p-r_i|| - ||p-r_j||)||²`
- **Solver**: Levenberg-Marquardt via `scipy.optimize.least_squares` with analytical Jacobian
- **Initialization**: Grid search over receiver-centered area to avoid local minima
- **Output**: Position + 2x2 covariance matrix + 95% confidence ellipse

```python
from pingu.locator.geometry import ReceiverGeometry
from pingu.locator.solvers import TDoASolver
from pingu.locator.posterior import position_uncertainty

geometry = ReceiverGeometry(receivers)
solver = TDoASolver(geometry)
estimate = solver.solve(tdoas, weights)  # → PositionEstimate

unc = position_uncertainty(estimate)
# {'semi_major': 5200.0, 'semi_minor': 3100.0, 'angle_degrees': 42.0, ...}
```

### Synthetic Data (`pingu.synthetic`)

Generates realistic test signals for all pipeline stages.

- **Signal generators**: SSB (Hilbert transform), CW (complex tone + keying), AM (carrier + envelope), FSK2/FSK4 (continuous-phase), BPSK/QPSK (RRC pulse-shaped)
- **Noise**: Calibrated complex AWGN at specified SNR
- **Scenarios**: Multi-receiver IQ with correct propagation delays (FFT phase-shift method)

```python
from pingu.synthetic.scenarios import TDoAScenario
from pingu.types import ModulationType

scenario = TDoAScenario(
    receivers=receivers, tx_position=(30_000, -20_000),
    sample_rate=48_000, snr_db=20.0, modulation=ModulationType.BPSK,
)
frames = scenario.generate()  # → dict[str, IQFrame]
```

### Visualization (`pingu.visualization`)

Plotting functions for analysis and debugging. All accept an optional `ax` parameter for subplot integration.

```python
from pingu.visualization.map_plot import plot_position_map
from pingu.visualization.convergence import plot_convergence

fig = plot_position_map(receivers, estimate=pos, true_pos=(30000, -20000))
fig.savefig("position.png")
```

### Pipeline (`pingu.pipeline`)

End-to-end orchestrator that wires all stages together.

```python
from pingu.config import load_config
from pingu.pipeline.runner import PinguPipeline

cfg = load_config("configs/default.yaml")
pipeline = PinguPipeline(config=cfg, receivers=receivers)

# Process frame-by-frame
estimate = pipeline.process_frame(frames)  # → PositionEstimate | None

# Or run until convergence
estimate = pipeline.run(list_of_frame_dicts)
```

## Configuration

All parameters are controlled via YAML configs loaded with OmegaConf:

```yaml
# configs/default.yaml (excerpt)
channelizer:
  method: polyphase       # polyphase | fft
  n_channels: 64

detector:
  pfa: 1.0e-6             # probability of false alarm

tdoa:
  method: phat            # phat | scot | ml
  interpolation: parabolic

integrator:
  process_noise: 1.0e-12  # very small for stationary target

locator:
  solver: levenberg_marquardt
  grid_search:
    enabled: true
    n_points: 50
    extent_km: 500
```

Override at runtime:

```python
cfg = load_config(overrides={"tdoa.method": "scot", "detector.pfa": 1e-4})
```

## Testing

```bash
uv run pytest                    # Run all 168 tests
uv run pytest tests/test_tdoa.py # Run a specific module
uv run pytest -v --tb=long       # Verbose output
uv run pytest --cov=pingu        # Coverage report
```

Test coverage spans:
- Signal generators (spectral properties, power normalization)
- Channelizer (tone placement, energy conservation, perfect reconstruction)
- Detector (CFAR detection, false alarm rate, SNR estimation)
- Classifier (CNN forward pass, dataset shapes, inference wrapper)
- TDoA (delay recovery within 1 sample at 20 dB SNR, all GCC methods)
- Kalman filter (convergence rate, 10x variance reduction in 100 updates)
- Position solver (recovery within 10 km for 100 km baseline geometry)
- CRLB (monotonicity with SNR, bandwidth, integration time)

## Technical Notes

- **Propagation**: Uses `c = 299,792,458 m/s`. Currently 2D line-of-sight; ionospheric hop geometry is a planned extension.
- **Precision**: Time/delay computations in `float64`. IQ samples in `complex64`.
- **GPU**: Training code checks for MPS (Apple Silicon) via `torch.backends.mps.is_available()`, falls back to CPU.
- **Clock sync**: Receivers assumed GPS-synchronized. `ReceiverConfig.clock_offset` models imperfections.

## Dependencies

**Core**: numpy, scipy | **ML**: torch, pytorch-lightning, scikit-learn | **Config**: omegaconf, pyyaml | **Viz**: matplotlib | **Dev**: pytest, pytest-cov, ruff

## License

This project is for research and educational purposes.
