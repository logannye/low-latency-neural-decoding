# Low-Latency Neural Decoding

[![Neural decoding](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/neural.yml/badge.svg)](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/neural.yml)
[![Python tests](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/test.yml/badge.svg)](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/test.yml)

Causal neural decoding under recording drift, limited calibration, and a 20 ms
processing schedule. The project connects decoding accuracy to an explicit
information budget, exported-model correctness, and measured streaming cost.
Protocols, source choices, failures and separate-code audit receipts are retained.

**Latest result: an adaptive calibration policy achieved mean held-out R²
0.187 [0.078, 0.299] using 33.75 seconds of fresh labels on average.** It reused
history without new labels on 17 of 24 selected sessions across eight known
participants. All sessions first require 30 seconds of unlabeled neural features.
The policy requested **71.9% fewer labels than 120-second daily calibration**, but
preserving that baseline's accuracy was **not established**. **T2 remains
unrecovered by the adaptive policy.**

![Adaptive decoding accuracy and calibration cost](neural_experiments/figures/v5-adaptive-calibration.png)

## A policy that can ask for more evidence

The controller first checks how historical models performed on earlier held-out
sessions. If that evidence supports reuse, it needs no fresh labels. Otherwise,
it requests 30, 60, 120, then at most 240 seconds of subsequent labels. Small
count-only and count-plus-power decoders compete using chronological prefix
validation, directional coverage, a time purge and comparisons with zero and
prefix-mean controls. A passing candidate is refitted; if none passes, the
controller abstains with zero output.

History is limited to three strictly earlier, layout-matched recordings from the
same person. The finite policy grid is selected with the target participant
excluded, before opening fresh assessment outcomes. This is startup calibration
in offline replay; ongoing drift-triggered recalibration is not implemented.

## Fresh results and information budgets

The V5 acquisition plan fixed archive ranks 13–15 before data access. Twenty-two
of 24 selected sessions had valid rows in the common 272–600 second score window.
Two T6 sessions had none and remain in cost accounting. Means weight sessions
equally within person, then weight eight people equally; 95% intervals resample
people. These are new sessions from known participants, not an untouched-person
transfer test. Archive order is not acquisition chronology.

| Procedure | Requested fresh labels, mean | Mean R² [95% interval] | Positive people | Angular error |
| --- | ---: | --- | ---: | ---: |
| Fixed historical prior | 0 s | 0.153 [0.034, 0.280] | 5/8 | 59.05° |
| Fixed prior + adaptation | 30 s | 0.195 [0.052, 0.331] | 6/8 | 54.69° |
| Daily scratch fit | 120 s | 0.202 [0.080, 0.323] | 6/8 | 53.85° |
| Daily scratch fit | 240 s | 0.274 [0.161, 0.385] | 8/8 | 50.13° |
| **Adaptive policy** | **33.75 s** | **0.187 [0.078, 0.299]** | **6/8** | **58.08°** |

Adaptive minus daily120 was **−0.015 [−0.073, 0.062]**, failing the predeclared
−0.05 noninferiority margin. Fixed history30 also remains competitive: slightly
higher accuracy with slightly fewer labels. Adaptive decoding passed the useful
cohort-accuracy gate, but does not dominate the fixed policies.

The adaptive policy accepted 20/22 scorable sessions, including two failures with
negative R². Both refusals and failed acceptances remain in the reported means.
The label is an offline cursor-to-target direction proxy, not measured limb
velocity or evidence of closed-loop control or clinical benefit.

## T2 and T3: failures retained

T2's exposed-data diagnosis found unreliable historical mappings, direction bias
in short calibration windows, and prefix selection that did not predict later
accuracy. Latest-session-only history and stronger low-label adaptation worsened
results. V5 tested conservative history reuse, scratch fallbacks and abstention.

On three fresh T2 sessions, the policy rejected zero-label reuse but still failed:
it accepted a 120-second model on day 214 with R² −0.032, then abstained on days 228
and 236. Its mean was **−0.022**. The fixed 240-second daily baseline reached
**0.057**, positive on all three days. More calibration can help, but this policy
did not reliably decide when to stop. T8 day 38, which lacked eligible history,
was the other false acceptance.

![Every fresh T2 result](neural_experiments/figures/v5-t2-transfer.png)

All seven available fresh T3 continuations abstained after 240 seconds. Their
mean R² **−0.084** equals the zero-output control; fixed daily120 reached −0.667.
Avoiding harmful predictions is useful behavior, but **T3 was not recovered and
its calibration time was not reduced**. One requested T3 session had no remaining
complete block; it was recorded without replacement.

## Engineering and verification

The decoder square-root transforms counts, optionally applies `log1p` to
spike-band power, causally smooths features and applies a ridge readout projected
to the unit disk. Filters reset at acquisition boundaries. Streaming inference
receives neural features only; state and per-step work grow linearly with channel
count. The controller keeps label access separate from the final scoring process.

Before fresh assessment, source choices were audited and committed. A separate
NumPy/scikit-learn reference reconstructed the fresh run's **186 full traces,
365 fits, 551 model records and 31 decisions**, including all metrics, bootstrap
intervals and gate outcomes. Maximum replay discrepancy was **1.47 × 10⁻¹⁴**.
The ten frozen execution sources stayed unchanged. This is separate-code
verification within the project, not external-laboratory replication.

Read the **[V5 report](neural_experiments/v5/REPORT.md)**,
[locked protocol](neural_experiments/v5/PROTOCOL.md),
[T2 diagnosis](neural_experiments/v5/DIAGNOSIS.md),
[data provenance](neural_experiments/v5/DATA.md), and
[reproduction guide](neural_experiments/v5/README.md).
The [complete aggregate results](neural_experiments/results/v5-study.json) retain
all cases, policy decisions, fixed gates, artifact hashes and the audit receipt.
[Validation evidence](neural_experiments/results/v5-validation.json) records the
software checks and retained audit attempts.

## Streaming measurement

The [V5 runtime evidence](neural_experiments/results/v5-runtime.json) measures the
predetermined T2 day 214 adaptive export on macOS ARM64. It retains this model's
negative accuracy result: runtime qualification does not establish decoding
quality. Three fast replays measured processing p99 of **0.019–0.020 ms**. Two
60-second runs paced at 20 ms measured processing p99 of **0.970–1.006 ms** and
release-to-completion p99 of **9.788–9.993 ms**, with **zero missed deadlines in
6,000 paced updates**. Numeric model arrays use **6,160 bytes** and persistent
numeric filter state uses **3,072 bytes**.

Measurements cover feature-ready processing and scheduling, with separate checks
of arithmetic and timestamps. The 200 ms filter's mean sample age is about 190 ms,
distinct from CPU latency. These results do not establish hard
real-time guarantees, acquisition latency, or concurrent calibration and streaming.

## Earlier evidence

The [V4 study](neural_experiments/v4/REPORT.md) established useful zero-fresh-label
history reuse on a different cohort, while T2 and T3 remained failures. The
[V3 study](neural_experiments/v3/REPORT.md) achieved 0.258 mean R² across nine known
participants and 26 different new sessions with 120-second daily calibration.
These cohorts and scoring windows differ from V5; headline means are not a
matched improvement comparison.

The [V1 report](neural_experiments/FINAL_REPORT.md) and
[V2 report](neural_experiments/v2/REPORT.md) preserve earlier temporal-network,
H1 and transfer experiments, including negative results. H1 predicts seven
movement coordinates and is not directly comparable to this direction proxy.

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
or native extension build is required. The [V5 guide](neural_experiments/v5/README.md)
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
