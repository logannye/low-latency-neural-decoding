# Low-Latency Neural Decoding

[![Neural decoding](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/neural.yml/badge.svg)](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/neural.yml)
[![Python tests](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/test.yml/badge.svg)](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/test.yml)

Causal neural decoding under recording drift, limited calibration, and a 20 ms
processing schedule. The project connects decoding accuracy to an explicit
information budget, exported-model correctness, and measured streaming cost.
Protocols, source choices, failures and independent audit receipts are retained.

**Latest result: 0.258 mean held-out R² across nine participants and 26 new
sessions**, with a participant-bootstrap 95% interval of **[0.069, 0.407]**.
The locked decoder uses **120 seconds of labeled daily calibration**. Eight of
nine participant means and 23 of 26 session scores are positive; mean angular
error is **51.44°**. It passes every predeclared offline accuracy criterion.

Given identical calibration labels and scoring windows, the original ridge
recipe reaches **0.021 R²**. The new procedure improves all nine participant
means, with paired gain **+0.237 R² [0.162, 0.310]**. These are held-out sessions
from previously studied people, using a cursor-to-target direction proxy.
They establish useful accuracy under the declared offline gate, not untouched
participant confirmation, zero-shot decoding, closed-loop control or clinical benefit.

![Held-out session accuracy by participant](neural_experiments/figures/v3-participant-accuracy.png)

## From failure diagnosis to a testable repair

The earlier learned temporal core fitted its calibration prefix but lost almost
all centered prediction–target correlation across days. Source validation was
already worse than a zero predictor; longer training increased overfitting.
Sparse channels, changing feature distributions and unconstrained output
magnitudes compounded the problem. Raw target, mask and channel-order audits
found no corresponding indexing error. The [failure analysis](neural_experiments/v3/DIAGNOSIS.md)
preserves these negative findings and their limits.

The replacement is deliberately compact: square-root spike counts plus optional
log spike-band power, causal exponential smoothing, prefix-only normalization,
strongly regularized ridge with a free intercept, and unit-disk output projection.
Every 20 ms, it updates bounded filter state and computes a two-coordinate command.
All observations advance state; acquisition boundaries reset it. Work and
persistent state grow linearly with channel count.

A finite 36-setting grid chooses the recipe using only other participants'
previously exposed sessions. All **3,528 development scores and 54 selections**
were independently checked and published before fresh assessment. Every
120-second primary selection chose counts plus power, a 200 ms smoothing time
constant, no slow-baseline subtraction, and mean-loss ridge lambda 1. Each new
session fits its own coefficients using only the permitted prefix.

| Procedure | Mean R² | Positive participant means | Mean angular error |
| --- | ---: | ---: | ---: |
| Zero vector | −0.034 | 0/9 | 90.00° |
| Calibration-prefix mean | −0.041 | 0/9 | 84.56° |
| Original ridge, daily refit | 0.021 | 6/9 | 59.49° |
| Selected count-only recipe | 0.259 | 7/9 | 53.77° |
| Locked counts + power recipe | **0.258** | **8/9** | **51.44°** |

The combined procedure improves some participants and angular error, but not cohort
mean R² over the separately selected count-only procedure. **T3 still fails**
(−0.373 R²), and T6 is weak (0.042). Reusing the new recipe's old-session mapping
without fresh labels also fails: **−0.734 mean R²**, with every participant mean
negative. Daily calibration changes the information available; the project does
not claim to have solved calibration-free recording drift.

On an identical later suffix, increasing calibration from 120 to 300 seconds
raises the selected procedure's R² from **0.216 to 0.312**. Both budgets have their
own source-selected settings, so this compares complete calibration procedures.

![Methods and matched calibration budgets](neural_experiments/figures/v3-methods-and-budget.png)

## Exact-export streaming cost

The protocol selected one representative export without consulting its score:
T2's first new archive-selected session, with 120-second calibration. On the
macOS ARM64 host it used **386 readout coefficients**, **6,160 bytes of numeric
model arrays** and **3,072 bytes of persistent numeric filter state**. These
counts exclude Python, scratch allocations and loaded replay data.

Across three fast replays, median processing p99 was **0.0158 ms**. In two
60-second runs paced at 20 ms, processing p99 was **0.622–0.662 ms** and
release-to-completion p99 was **5.10–5.13 ms**. No 20 ms deadline misses were
observed in 6,000 paced steps. Maximum processing time reached 2.92 ms;
the 2 ms target concerns p99, not every observation. Independent timestamp
audit passed, and full batch/stream discrepancy was at most **3.17 × 10⁻¹⁵**.

The selected 200 ms causal smoothing filter has approximately **190 ms mean
sample age**, a separate response property rather than a fixed end-to-end
delay. These are feature-ready measurements of one export, not a hard real-time
guarantee or timings for every participant. See the
[runtime evidence](neural_experiments/results/v3-runtime.json).

## Evidence and engineering

- **Evaluation boundaries:** locked archive-member selection, disjoint new sessions,
  participant-excluded source ranking, equal-person summaries, fixed scoring masks,
  matched calibration budgets and explicit unavailable cases.
- **Inspectable computation:** closed-form fitting with a documented penalty scale,
  prefix-only statistics, validation before state mutation, strict numeric exports,
  feature-only inference and recording-boundary resets.
- **Independent verification:** 361 models and 722 scored traces checked; every
  nonconstant assessment model replayed independently, with maximum prediction
  discrepancy **1.41 × 10⁻¹⁴**. Separate checks reconstruct ridge optimality,
  scores, bootstrap intervals, gate decisions and runtime timestamps.
- **Reproduction:** bounded public-data acquisition, preserved source snapshots and
  hashes, synthetic leakage/tamper tests, Python CI, and aggregate-only publication.
  The historical neural-network and CNN experiments remain reproducible.

Read the **[V3 results report](neural_experiments/v3/REPORT.md)**,
[locked protocol](neural_experiments/v3/PROTOCOL.md),
[data provenance](neural_experiments/v3/DATA.md), and
[reproduction guide](neural_experiments/v3/README.md).
The [aggregate results](neural_experiments/results/v3-results.json) contain
participant/session scores, exact identities and the independent audit receipt.

## Earlier results remain available

The original H1 TCN averaged **0.1776 R²** on seven held-out recording days versus
0.0341 for ridge. A later 30-second supervised correction improved H1 development
R² from **0.174 to 0.298** on a common suffix, winning all nine day/seed comparisons.
H1 predicts seven movement coordinates, so its values are not comparable to this
nine-person direction-proxy task. Read the [V1 report](neural_experiments/FINAL_REPORT.md)
and [V2 report](neural_experiments/v2/REPORT.md).

The V2 frozen-core transfer result remains explicitly negative: mean future-day
R² **−1.591**, despite outperforming scratch. Random frozen features performed
better than the learned core. V2's same-export NumPy implementation achieved
0.029–0.030 ms median fast-replay p99 and 0.872–1.311 ms paced processing p99;
those timings belong to the V2 artifacts and are separate from V3 evidence.

## Quick start

Use Python 3.12 and prebuilt scientific libraries:

```sh
python -m pip install --only-binary=:all: -e '.[dev,neural,runtime,benchmark]'
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m pytest -q tests/neural_decoder
neural-decode --help
python neural_experiments/replay_demo.py --smoke --output work/replay-smoke.html
```

Synthetic tests and the smoke replay require no neural-data download. No Rust
or native extension build is required. The [V3 guide](neural_experiments/v3/README.md)
covers acquisition, source selection, one-shot assessment, independent audit,
plots and timing. Raw recordings, prepared arrays, models, coefficients and
sample-level predictions remain in ignored local caches.

Runtime begins when neural features are available. Upstream feature-extraction
causality and acquisition/transport latency remain unverified. Causal smoothing
uses no future samples, but its response time is distinct from CPU latency.

## Earlier image-classification work

The original CNN experiment is preserved in [CNN_EXPERIMENT.md](CNN_EXPERIMENT.md),
with its source, `cnn-*` commands, and
[matched Python/ARM64 results](experiments/baselines/cifar10-matched-cpu-25-v1/README.md).
The repository now focuses on neural decoding, adaptation and streaming inference.
