# H1 causal decoding protocol, development version 1

Date: 2026-10-22. Starting repository revision: `6f0cfd932b41503fed3899211491bacfbfba95e7`.
This experiment is separate from all historical CNN results. The next gates cover a bounded adaptation hypothesis, runtime evaluation, and
sealed-day assessment. No final model or adaptation recipe is frozen yet.

## Information and data roles

FALCON H1 only, upstream source `770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f`.
Each provider calibration file is a separate stream. Preserve all bins and reset
only at file boundaries; do not use trial labels, masks, targets, or future bins
in inference. The local acquisition manifest pins mutable draft assets by UUID,
provider digest and SHA256. Raw data and derived arrays stay local because the
provider data-use agreement prohibits redistribution. Do not commit them.

Development folds: S0-S1 -> S2; S0-S2 -> S3; S0-S3 -> S4.
S5 is sealed until recipe freeze. S6-S12 calibration recordings are sealed local
evaluation, not new-day training or development. Schema-only inspection is allowed.
Minival is excluded from model fitting and independent evidence because it overlaps
calibration. No hidden evaluation or external submission is part of this session.

## Prespecified baseline recipe

See `configs/ridge-v1.json`, written before development scores. Six ridge
configurations: history 1 or 5 bins and lambda 0.001, 0.01, or 0.1.
For each channel, first apply a 12-tap exponential FIR with weights proportional to
`exp(-k/12)`, k=0..11 (240 ms time constant, 20 ms bins), with zero input before
file start. Then center and scale using all training neural bins only. Standard
deviations are floored at 1e-6. Concatenate the current and previous standardized
features; unavailable history at file start is zero in standardized space.
Five lagged filtered bins span 16 raw bins (320 ms total receptive field).

Fit an unpenalized intercept and minimize `mean((XW+b-Y)^2) + lambda*sum(W^2)`
per output using supplied training loss masks. No target-defined stillness exclusion.
The constant baseline predicts the training masked-target mean. Trials and unscored
evaluation bins continue to advance state. No per-file evaluation normalization.
Select largest mean development-day variance-weighted R2; ties prefer shorter
history, then larger regularization. These are development scores used for selection.

## Scoring and evidence

Primary decoding metric: mean of the per-day variance-weighted R2 values, concatenating
files within each day before centering targets. Keep each dimension, each day, worst
day, mask sample count, and official population standard deviation. Negative R2 is
retained. Report absolute paired R2 differences. No frame bootstrap or uncertainty
claim from three development days. Numerical parity to pinned official helper and
wrapper is tested independently using synthetic unequal-length S6/S10 fixtures.

Exploratory runs use fresh ignored directories containing source snapshots, source
and input hashes, environment versions, configuration, numeric model artifacts,
predictions and completion/failure state. Summary tables may be tracked. Final
performance comparisons require a committed source and frozen recipe later.

## Next gates

G1 requires actual development baseline scores and all contract checks. G2 adds one
23,367-parameter TCN: three single k=3 convolutions, width32, dilations1/2/4,
ReLU and linear seven-output head, 15-bin causal receptive field. Choose it for
an interpretable finite history and a same-weight incremental execution comparison.
Training settings and a bounded pilot are declared before that model is scored.

TCN recipe `configs/tcn-v1.json`: raw counts standardized using all training
observations; targets standardized using masked training observations. Normalization
arithmetic, model weights and inference are float32. Adam lr=0.001, no weight decay,
20 fixed epochs, last checkpoint, seeds0/1/2, one CPU thread. Each training chunk
contains 14 preceding real bins and up to256 loss positions from the same file;
batch16 chunks, shuffle chunks per epoch. Exclude first14 bins of each file from
training loss only; evaluate every supplied scored bin including file startup.
Right padding has no loss. No evaluation-based early stopping or best-seed choice.
Pilot: one epoch foldA seed0 for time only; it produces no development score.

One robustness hypothesis must follow baseline diagnostics. Do not run adaptive
normalization or inspect sealed data merely because code is available. The optional
default is source-anchored bounded normalization, with causal predict-then-update
ordering, explicit settings, and observation-nuisance versus task-mixture tests.

Runtime goals remain provisional: p99 complete counts-ready step <2 ms. Short
correctness traces or training timings are not sustained runtime measurements.
