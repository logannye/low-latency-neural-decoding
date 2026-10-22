# Low-Latency Neural Decoding

[![Neural decoding](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/neural.yml/badge.svg)](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/neural.yml)
[![Python tests](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/test.yml/badge.svg)](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/test.yml)

Causal neural decoding under recording drift, limited calibration, and a 20 ms
processing schedule. This project connects compact temporal models, supervised
adaptation, participant-held-out evaluation, and bounded streaming execution.
Every reported result is tied to a fixed protocol, data and model identities,
and retained predictions that an independent scorer can check.

**Accuracy:** a 30-second supervised readout correction improved H1 development
R² from **0.174 to 0.298** on a common suffix, winning all nine day/seed comparisons.
The correction folds into the existing **23,367-parameter** model without adding
streaming operations. Ten-second calibration performed worse; larger-context
and loss-weighting variants failed their selection gate.

![H1 drift and calibration](neural_experiments/figures/v2-h1-calibration.png)

**Beyond one participant:** the new study uses **nine people and 54 sessions**
from the BrainGate array-performance release. A participant-specific input
adapter feeds a shared causal temporal core. Each outer fold excludes the entire
target person from pretraining and source checkpoint selection, then compares
frozen-core transfer, fine-tuning, the same architecture from scratch, and causal
ridge at identical 30/120/300-second target-label budgets.

**Transfer result and limitation:** at the prespecified 120-second budget,
frozen-core transfer beat scratch in **8 of 9 people**: mean paired gain
**+1.518 R²**, participant-bootstrap 95% interval **[+0.495, +2.715]**. However,
absolute future-day means were **−1.591 for transfer and −3.108 for scratch**,
and every person's transfer mean was negative. This is relative protection
against a failing baseline; useful cross-person decoding is **not established**.
All 260 models and 1,560 traces passed the independent protocol/scoring audit.

A subsequent, explicitly **post-hoc freezing control** was decisive: random
frozen features beat the learned frozen core in **all nine people**, averaging
**−0.562 R²**. Thus the primary gain over scratch cannot be credited to source
pretraining in this experiment. All absolute future-day person means remain
negative. The control's 27 models and 162 traces passed a separate verification
pass using the existing scikit-learn scoring reference.

![Participant-held-out transfer results](neural_experiments/figures/v2-participant-transfer.png)

H1 predicts seven movement coordinates; the nine-person study predicts a
cursor-to-target direction proxy. Their R² values measure different tasks.
H1 V2 is development evidence on previously exposed days. The new participant
study tests calibrated transfer, with target labels explicitly budgeted. Neither
study establishes zero-shot decoding, closed-loop control, or clinical benefit.

**Systems:** exact exported models achieved **0.029–0.030 ms median fast-replay
p99** with incremental NumPy, **3.12× lower for calibrated H1 and 5.88× lower
for the transfer model** than the matching PyTorch window implementations. NumPy
paced processing p99 was **0.872–1.311 ms**, below the 2 ms target, with no observed
20 ms deadline misses. Scheduler delay brought measured release-to-completion
p99 to approximately **9.9–10.0 ms**. These are feature-ready replay timings,
with processing and scheduling costs reported separately.

![Same-weight streaming runtime](neural_experiments/figures/v2-transfer-runtime.png)

## What the repository demonstrates

- **Experimental control:** matched-exposure drift diagnostics, common training masks and evaluation suffixes, source-only checkpoint selection, three seeds, and a prespecified participant-level primary contrast.
- **Model design:** compact causal TCNs, participant-specific adapters for incompatible electrode spaces, residual temporal cores, and closed-form readout correction.
- **Streaming engineering:** exact-weight NumPy/PyTorch parity, bounded activation history, recording-boundary resets, validation before state mutation, and complete-step fast/paced measurements.
- **Evidence:** separate feature and scorer files; hashes for plans, source snapshots, normalizers, exports, masks, and traces; independent score and timestamp audits; synthetic tests that attack leakage and artifact changes.
- **Reproducibility:** bounded public-data acquisition, explicit unavailable cases, aggregate-only publication, and preservation of negative results.

Read the **[V2 report](neural_experiments/v2/REPORT.md)**,
[locked protocol](neural_experiments/v2/PROTOCOL.md),
[H1 accuracy results](neural_experiments/v2/ACCURACY.md),
[cohort provenance](neural_experiments/v2/DATA.md), and
[aggregate evidence](neural_experiments/results/). The
[reproduction guide](neural_experiments/v2/README.md) covers acquisition,
training, independent audits, head composition, and timing.

## Original held-out result

The frozen V1 H1 model averaged **0.1776 R²** on seven later recording days,
versus **0.0341** for ridge, with no new-day labels. Every one of 21 day/seed
comparisons favored the TCN. Its separate development-weight systems experiment
measured **0.034 ms median fast-replay p99** for incremental NumPy, about **3.13×**
lower than PyTorch; paced p99 was **0.428–0.538 ms** on an M4 Max.

Those original results remain in the [V1 report](neural_experiments/FINAL_REPORT.md).
They use one participant's public FALCON calibration recordings and are not
FALCON leaderboard scores. V1 timing and final quality used separately pinned
weights. ONNX recorded one deadline miss; the full implementation/repeat record
is retained. V2 measures its selected exported artifacts separately.

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
or native extension build is required. Raw recordings, prepared arrays, model
weights, calibration coefficients, and sample-level predictions remain in
ignored local caches. Public artifacts contain code, protocols, metadata,
aggregate scores, and plots. Runtime begins when supplied neural features are
available; upstream feature-extraction causality and neural-acquisition latency
remain unverified.

## Earlier work

The original image-classification experiment is preserved in
[CNN_EXPERIMENT.md](CNN_EXPERIMENT.md), with its source, `cnn-*` commands, and
[matched Python/ARM64 results](experiments/baselines/cifar10-matched-cpu-25-v1/README.md).
The repository's primary direction is now neural decoding, adaptation, and
streaming inference.
