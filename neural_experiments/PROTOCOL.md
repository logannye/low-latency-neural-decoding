# H1 causal decoding protocol, version 1

Date: 2026-10-22. Starting repository revision: `6f0cfd932b41503fed3899211491bacfbfba95e7`.
This experiment is separate from all historical CNN results. Development is complete.
The selected recipe is `configs/frozen-recipe-v1.json`: constant and selected ridge
baselines plus all three seeds of the fixed TCN with training channel dropout 0.1.
Frozen file manifests bind fitted weights and committed source before sealed release.

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
calibration. Official hidden evaluation requires a separate compatible service/package and access;
local public-recording scores must never be called leaderboard scores.

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

## Model and final assessment contract

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

## Frozen assessment and optional calibration

G3 compared frozen, ordinary EMA, and three bounded corrections with fixed
30-second time constant. It failed the prespecified bound-retention rule; retain
frozen normalization. The second/final hypothesis added training-only per-chunk
channel dropout p=0.1. It passed its prespecified mean-gain and day-harm rules;
all other TCN training settings remain fixed. No seed was selected for quality.
See ADAPTATION.md and AUGMENTATION.md for every result and limitation.

Fit the selected model set on S0-S4, freeze exact weights/source/recipe/metrics,
and then evaluate S5 once. Refit that identical recipe on S0-S5 without
redesign, create a child freeze with matching recipe/source identities, and
evaluate all14 public calibration files on S6-S12. Report every seed/day,
per-output score, sample count, duration, worst day, and population day SD.
The early result is one independent day; the later result has seven days from
the same participant. No frame-level confidence interval is justified.

The optional supervised study runs only after primary evaluation. Use a separate
job to fit a seven-output intercept correction from each file's first0/5/15/30
seconds of scored labels. All arms use identical frozen predictions and inference
state from the entire preceding neural sequence, reset per file. Every arm scores
the same suffix beginning at bin1500 (30seconds), with its original mask. No
label from another or later file can enter calibration. Report nominal and actual
scored prefix seconds per file/day. This is continuous-prefix calibration, not
the official challenge's disjoint calibration-to-hidden-evaluation protocol.

G4 uses float32 same-weight PyTorch window, NumPy incremental, and CPU ONNX
explicit-state execution, one thread, 500warmup bins, full-file fast replay and
120seconds paced replay per implementation in counterbalanced blocks. Retain
raw monotonic stamps, queue/deadline and memory/CPU accounting. Strict trace
and score parity precede timing. Compression is a separately identified
approximation with no seed/day quality loss beyond0.005 allowed for acceptance.
