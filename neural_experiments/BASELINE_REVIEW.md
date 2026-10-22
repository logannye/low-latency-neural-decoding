# Independent review of the first H1 baseline run

The development result bundle `results/neural-h1/ridge-dev-v1` passed an
independent numerical audit on October 22, 2026. All 21 fitted artifacts and 42
full prediction traces were checked. The selected ridge model uses one filtered
bin and regularization 0.1. Its development-day mean R² is **0.0526246**, with a
negative score on S2. These are local development results on public FALCON
calibration recordings, used for model selection. S5 and S6–S12 remain sealed;
this review opened no sample arrays from those days.

| Model | S2 | S3 | S4 | Equal-day mean |
|---|---:|---:|---:|---:|
| Training target mean | -0.0000531 | -0.0000276 | -0.0000300 | -0.0000369 |
| Selected causal ridge | -0.0728941 | 0.0995541 | 0.1312138 | 0.0526246 |

The ridge selection compared six settings: histories 1 and 5, each at penalties
0.001, 0.01, and 0.1. Every candidate and every day is retained in the machine
readable summary and audit. Five-bin history did not win this development grid.
S2 is not dropped or clipped from the reported mean. With only three development
days and settings chosen on those days, the mean is not an unbiased estimate of
final performance.

## What the independent computation checked

`neural_experiments/audit.py` uses NumPy convolution and explicit lag matrices,
without importing production feature transforms, model predictors, or scoring
functions. It reconstructed all full prediction traces from the saved models
and features. The maximum absolute difference was **5.8981e-17**. It recalculated
per-output and variance-weighted day R², equal-day means, population standard
deviations, and the selection decision. It independently refit the selected
ridge setting on each fold and checked its coefficients and intercept.

The audit verified the retained source snapshot and diff, configuration, copied
input index, saved models, prediction files, and prepared feature/target arrays
against their recorded SHA-256 values. It verified the raw-file hashes for the
11 development files only. Before loading sample arrays, `data.load_index`
reconstructed asset/day/provider-split identities from the source manifest and
the pinned official session mapping. The frozen forward folds are S0–S1 → S2,
S0–S2 → S3, and S0–S3 → S4. No evaluation day's labels or neural observations
entered its training normalizer or ridge fit.

The final machine-readable review is
`results/neural-h1/ridge-dev-v1/independent-audit-v2.json`. Its exact audit source
is retained beside it as `independent-audit-source-v2.py`; the initial audit and
source are also preserved. Reproduce into a new file:

```sh
PYTHONPATH=src .venv/bin/python neural_experiments/audit.py \
  --run results/neural-h1/ridge-dev-v1 \
  --index data/neural_h1/prepared/index.json \
  --output results/neural-h1/ridge-dev-v1/independent-audit-new.json
```

This is an exploratory source snapshot rooted at Git revision
`6f0cfd932b41503fed3899211491bacfbfba95e7`, with new code retained by exact file
hashes. It is not a claim that this historical commit already contained the new
experiment. Final measured comparisons still require a committed reproducible
source state. No runtime performance measurement is certified by this review.

## Causal and implementation review

The filter is a normalized, truncated exponential FIR with 12 taps and a
12-bin time constant. Each output uses current and preceding bins; raw input is
zero-padded at each file start. Standardization uses all neural bins from the
authorized training files. Regression fitting uses only their declared loss-mask
rows. The history vector places the current filtered/standardized bin first,
then older bins; unavailable standardized history at file start is zero.

A one-bin feature model therefore already has 12 raw input bins of causal
support, or a 240 ms collection window at the nominal bin period. Five-bin
history extends that support to 16 raw bins, or 320 ms. Neither has future
lookahead. Smoothing still introduces response lag; these widths are not
processing latency measurements. Batch training features and independent
streaming reconstruction matched at file boundaries. Predictor state continues
across trials and is reset for every provider file.

Two implementation issues were identified and corrected in the current source
after this run; the original run source snapshot was retained:

- The initial training index checker trusted self-declared day fields without
  reconstructing their source-asset mapping. It now invokes `data.load_index`
  before reading sample files. The independent audit establishes that the
  already-completed run used the correct development assets.
- The initial replay code converted sequence IDs with `int(seq)`, allowing
  fractional IDs to be silently truncated. Replay now uses the shared feature
  loader, which checks the full sequence against consecutive IDs before replay.

The original fitted models and measured traces are numerically validated by the
independent audit; these gate corrections do not turn the old run into a new
experiment or alter its recorded source identity. Root integration tests own
the verification of these corrected ingestion paths.

## Development failure diagnostic

For the selected model, residual error was decomposed as
`SSE = N * mean(target - prediction)^2 + sum(centered_residual^2)` separately for
each output. The identity was checked numerically. The bias term explains a
larger share of S2 error than of S3 or S4 error.

| Day | Scored bins | Mean residual bias share of total SSE | Input mean shift RMS, in training SD |
|---|---:|---:|---:|
| S2 | 22,790 | 18.19% | 0.870 |
| S3 | 21,746 | 4.82% | 1.052 |
| S4 | 23,368 | 5.58% | 0.728 |

S2's seventh output (zero-based index 6) had R² -0.5303; its mean residual
contributed 46.19% of that output's squared error and most of the day's bias
error. The output is identified by position because anatomical names are not
established by this review. Its residual mean is a retrospective diagnostic
computed using evaluator-only labels, not a quantity available to the live
predictor.

These observations justify testing whether a bounded correction can reduce
some offset-related error. They do not identify recording drift as its cause.
S3 has a larger overall neural mean shift than S2 but better decoding, so a
single shift-magnitude statistic does not explain the failing day. A change in
movement mixture could also move the neural distribution. A causal adaptation
experiment must therefore include frozen and ordinary online-normalization
comparators, report days harmed, and test a changed movement mixture as well as
an imposed observation nuisance. No adaptation benefit is claimed here.
