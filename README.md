# PINGU — HF TDoA Geolocation Pipeline

PINGU is a hybrid traditional/ML signal processing pipeline for geolocating stationary HF radio transmitters using **Time Difference of Arrival (TDoA)** measurements across distributed receivers.

The pipeline uses traditional DSP for channelization, cross-correlation, Bayesian filtering, and non-linear position solving, while reserving ML for signal detection and modulation classification where synthetic training data is practical.

## Quick Start

```bash
# Requirements: Python 3.11+, uv
uv sync --extra dev

# Run tests (219 unit tests)
uv run pytest

# Run the pipeline on synthetic scenarios
uv run python scripts/run_scenarios.py --modulations cw bpsk qpsk --snr 20 --n-frames 20

# Run all modulations with per-frame tracing
uv run python scripts/run_scenarios.py --modulations all --snr 20 --n-frames 20 --trace

# SNR sweep with plots
uv run python scripts/run_scenarios.py --modulations all --snr-start 0 --snr-stop 30 --snr-step 10 --plot

# Generate synthetic training data
uv run python scripts/generate_synthetic.py --n-scenarios 100 --output data/synthetic/

# Train the modulation classifier
uv run python scripts/train_amc.py --epochs 50
```

## Performance at 2 Msps (20 dB SNR)

All modulations converge within 5 frames at 20 dB SNR:

| Modulation | Position Error | Mean TDoA Error |
|------------|---------------|-----------------|
| SSB        | 0.1 m         | 0.00 us         |
| FSK4       | 0.9 m         | 0.00 us         |
| FSK2       | 1.8 m         | 0.01 us         |
| QPSK       | 4.1 m         | 0.02 us         |
| BPSK       | 5.4 m         | 0.02 us         |
| AM         | 97.0 m        | 0.50 us         |
| CW         | 163.5 m       | 0.68 us         |

## How It Works

The pipeline processes wideband IQ data (2 Msps) from 5 GPS-synchronized receivers through six stages:

```
Wideband IQ (2 Msps)
       |
       v
  Channelize (64 ch, 31.25 kHz each)
       |
       v
  Detect signal in channel X (CFAR)
       |
       v
  Bandpass filter ORIGINAL wideband IQ to channel X band
  (keeps full 2 Msps sample rate, rejects 63/64 channels of noise)
       |
       v
  GCC-basic on filtered signals -> TDoA (0.5 us resolution)
       |
       v
  Kalman filter -> Converge -> NLLS Position Solve
```

1. **Channelize** — Split wideband IQ into 64 narrowband channels (polyphase filterbank or FFT)
2. **Detect** — CFAR energy detector identifies active channels
3. **Classify** — CNN classifies modulation type (SSB, CW, AM, FSK, PSK)
4. **Bandpass + TDoA** — Bandpass filter wideband IQ to detected channel, then correlate at full sample rate using GCC-basic for unambiguous delay estimation
5. **Integrate** — Kalman filter fuses TDoA estimates over time (stationary target assumption reduces variance as 1/N)
6. **Locate** — Weighted non-linear least squares (Levenberg-Marquardt) solves for transmitter position with grid-search initialization

### Why Bandpass-then-Correlate?

The channelizer decimates to `fs/M` (31.25 kHz), destroying time resolution. Correlating full wideband IQ directly drowns the signal in noise from 63 empty channels. The bandpass approach gives both:

- **Noise rejection**: removes 63/64 channels of noise (~20 dB improvement)
- **Full time resolution**: 2 Msps = 0.5 us = 150 m per sample
- **GCC-basic** is used after filtering because PHAT/SCOT would amplify stopband noise in the mostly-empty spectrum

## Pipeline Tracing

The `--trace` flag provides per-frame diagnostics:

```
Frame | Dets |     Ch |   GCC |  Mean TDoA Err | Kalman Var |             Position | Solver
-------------------------------------------------------------------------------------------
    1 |   15 | [0, 1, | basic |         0.02 us |   6.75e-24 |                    - |      -
    2 |   15 | [0, 1, | basic |         0.02 us |   6.75e-24 |                    - |      -
    5 |   19 | [0, 1, | basic |         0.02 us |   6.74e-24 |      (29998, -19996) |     OK
```

Each frame shows: detection count, selected channels, GCC method, TDoA error vs ground truth, Kalman variance, and solver result.

## Project Structure

```
PINGU/
├── pyproject.toml                    # Dependencies and build config
├── configs/
│   ├── default.yaml                  # Master configuration (2 Msps)
│   ├── scenarios/example_sweep.yaml  # Multi-modulation SNR sweep
│   ├── training/amc_baseline.yaml    # AMC CNN training hyperparameters
│   └── geometry/pentagon_5rx.yaml    # 5-receiver layout preset
│
├── src/pingu/
│   ├── types.py              # Shared dataclasses (IQFrame, FrameTrace, etc.)
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
│   │   ├── gcc.py            #   GCC-PHAT, GCC-SCOT, GCC-ML, GCC-basic
│   │   ├── peak_interpolation.py  # Parabolic + sinc sub-sample refinement
│   │   ├── pair_manager.py   #   Manages C(n,2) = 10 receiver pairs
│   │   └── uncertainty.py    #   Cramer-Rao Lower Bound
│   │
│   ├── integrator/           # Stage 4: Bayesian temporal filtering
│   │   ├── kalman.py         #   Kalman filter (returns innovation vector)
│   │   └── convergence.py    #   Variance tracking + CRLB comparison
│   │
│   ├── locator/              # Stage 5: Position estimation
│   │   ├── geometry.py       #   Receiver positions + TDoA geometry
│   │   ├── cost_functions.py #   Weighted NLLS residuals + Jacobian
│   │   ├── solvers.py        #   Levenberg-Marquardt solver
│   │   └── posterior.py      #   Confidence ellipses
│   │
│   ├── synthetic/            # Data generation
│   │   ├── signals.py        #   SSB, CW, AM, FSK2/4, BPSK, QPSK, NOISE
│   │   ├── noise.py          #   Calibrated AWGN
│   │   ├── scenarios.py      #   Multi-receiver TDoA scenarios
│   │   └── channels.py       #   Watterson channel model (placeholder)
│   │
│   ├── pipeline/
│   │   └── runner.py         #   End-to-end orchestrator with bandpass TDoA
│   │
│   ├── scenarios/            # Batch scenario execution
│   │   ├── spec.py           #   ScenarioSpec + sweep expansion
│   │   ├── runner.py         #   ScenarioRunner with true TDoA comparison
│   │   └── report.py         #   Tables, stats, CSV, traces, plots
│   │
│   └── visualization/
│       ├── spectrogram.py    #   IQ spectrogram plots
│       ├── correlation.py    #   Cross-correlation plots
│       ├── map_plot.py       #   Position map + confidence ellipses
│       └── convergence.py    #   Variance-over-time plots
│
├── tests/                    # 219 unit tests (mirrors src/ structure)
├── scripts/
│   ├── run_scenarios.py      #   Run multi-scenario sweeps with --trace
│   ├── run_pipeline.py       #   Run pipeline on a single scenario
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
| `FrameTrace` | Per-frame diagnostic trace (detections, TDoA errors, Kalman state) |
| `ReceiverConfig` | Receiver location (lat/lon or Cartesian) and parameters |
| `ModulationType` | Enum: SSB, CW, AM, FSK2, FSK4, BPSK, QPSK, NOISE |

### Channelizer (`pingu.channelizer`)

Splits wideband IQ into narrowband channels for per-signal processing.

- **PolyphaseChannelizer** — M-path polyphase filterbank with configurable overlap factor and prototype filter. Near-perfect reconstruction. This is the primary channelizer.
- **FFTChannelizer** — Simpler overlap-save implementation for reference and validation.

```python
from pingu.channelizer.polyphase import PolyphaseChannelizer

ch = PolyphaseChannelizer(n_channels=64, overlap_factor=4)
result = ch.channelize(iq_frame)  # -> ChannelizedFrame
# result.channels: shape (64, n_samples_per_channel)
```

### Detector (`pingu.detector`)

CFAR (Constant False Alarm Rate) energy detector scans channelized frames for active signals. Supports both time-domain and frequency-domain CFAR modes.

```python
from pingu.detector.energy import EnergyDetector

det = EnergyDetector(pfa=1e-6, block_size=32768, guard_cells=4, reference_cells=16)
detections = det.detect(channelized_frame)  # -> list[Detection]
```

### TDoA Estimation (`pingu.tdoa`)

Estimates time differences of arrival between all receiver pairs using generalized cross-correlation.

- **GCC-basic** — Unweighted cross-correlation (used after bandpass filtering)
- **GCC-PHAT** — Phase transform weighting (best for broadband signals in wideband)
- **GCC-SCOT** — Smoothed coherence transform
- **GCC-ML** — Maximum-likelihood weighting
- **Sub-sample interpolation** — Parabolic (fast) or sinc (accurate) refinement
- **CRLB** — Cramer-Rao Lower Bound for variance estimation

```python
from pingu.tdoa.pair_manager import PairManager

pm = PairManager(["RX0", "RX1", "RX2", "RX3", "RX4"])  # 10 pairs
tdoas = pm.estimate_all_tdoas(signals, fs=2_000_000, method="basic",
                               max_delay_samples=10000)
```

### Pipeline (`pingu.pipeline`)

End-to-end orchestrator with bandpass-then-TDoA architecture and per-frame tracing.

```python
from pingu.config import load_config
from pingu.pipeline.runner import PinguPipeline

cfg = load_config("configs/default.yaml")
pipeline = PinguPipeline(config=cfg, receivers=receivers)

# Process frame-by-frame
estimate = pipeline.process_frame(frames)  # -> PositionEstimate | None

# Or run until convergence
estimate = pipeline.run(list_of_frame_dicts)

# Access per-frame diagnostics
for trace in pipeline.traces:
    print(f"Frame {trace.frame_index}: TDoA err = {trace.tdoa_delays_s}")
```

### Scenario Runner (`pingu.scenarios`)

Batch execution engine for parameter sweeps with ground-truth comparison.

```python
from pingu.scenarios import ScenarioRunner, ScenarioReport, specs_from_cli_args

runner = ScenarioRunner(config=cfg)
results = runner.run_all(specs)  # Each result includes traces with true TDoAs
report = ScenarioReport(results)
report.summary_table()
report.print_frame_traces(scenario_index=0)
report.plot_error_vs_snr(save_path="error_vs_snr.png")
```

### Synthetic Data (`pingu.synthetic`)

Generates realistic test signals for all pipeline stages. Signal parameters auto-scale with sample rate.

- **Signal generators**: SSB, CW, AM, FSK2/FSK4, BPSK, QPSK, NOISE — all with sample-rate-aware parameter scaling
- **Noise**: Calibrated complex AWGN at specified SNR
- **Scenarios**: Multi-receiver IQ with correct propagation delays (FFT phase-shift method)

```python
from pingu.synthetic.scenarios import TDoAScenario
from pingu.types import ModulationType

scenario = TDoAScenario(
    receivers=receivers, tx_position=(30_000, -20_000),
    sample_rate=2_000_000, snr_db=20.0, modulation=ModulationType.BPSK,
)
frames = scenario.generate()  # -> dict[str, IQFrame]
```

## Configuration

All parameters are controlled via YAML configs loaded with OmegaConf:

```yaml
# configs/default.yaml (excerpt)
receivers:
  sample_rate: 2000000.0      # 2 Msps wideband

channelizer:
  method: polyphase
  n_channels: 64              # 31.25 kHz per channel
  overlap_factor: 4

detector:
  pfa: 1.0e-6
  block_size: 32768           # ~16 ms integration at 2 Msps

tdoa:
  method: auto                # auto-selects GCC method per modulation
  fft_size: 131072
  max_delay_samples: 10000    # 5 ms = 1500 km range

integrator:
  process_noise: 1.0e-12
```

Override at runtime:

```python
cfg = load_config(overrides={"tdoa.method": "basic", "detector.pfa": 1e-4})
```

## Testing

```bash
uv run pytest                    # Run all 219 tests
uv run pytest tests/test_tdoa.py # Run a specific module
uv run pytest -v --tb=long       # Verbose output
uv run pytest --cov=pingu        # Coverage report
```

Test coverage spans:
- Signal generators (spectral properties, power normalization, sample-rate scaling)
- Channelizer (tone placement, energy conservation, perfect reconstruction)
- Detector (CFAR detection, false alarm rate, SNR estimation, DOF handling)
- Classifier (CNN forward pass, dataset shapes, inference wrapper)
- TDoA (delay recovery within 1 sample at 20 dB SNR, all GCC methods)
- Kalman filter (convergence rate, innovation vectors, 10x variance reduction)
- Position solver (recovery within 10 km for 100 km baseline geometry)
- Scenario runner (batch execution, tracing, report generation)
- CRLB (monotonicity with SNR, bandwidth, integration time)

## Technical Notes

- **Sample rate**: Default 2 Msps (0.5 us time resolution = 150 m position resolution)
- **Propagation**: Uses `c = 299,792,458 m/s`. Currently 2D line-of-sight; ionospheric hop geometry is a planned extension.
- **Precision**: Time/delay computations in `float64`. IQ samples in `complex64`.
- **GPU**: Training code checks for MPS (Apple Silicon) via `torch.backends.mps.is_available()`, falls back to CPU.
- **Clock sync**: Receivers assumed GPS-synchronized. `ReceiverConfig.clock_offset` models imperfections.

## Dependencies

**Core**: numpy, scipy | **ML**: torch, pytorch-lightning, scikit-learn | **Config**: omegaconf, pyyaml | **Viz**: matplotlib | **Dev**: pytest, pytest-cov, ruff

## License

This project is for research and educational purposes.
