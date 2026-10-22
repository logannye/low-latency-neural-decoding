# V4: history priors, reduced calibration and T3 continuation

V4 tests whether the decoder can reuse earlier recordings to reduce fresh labels,
and whether targeted regularization and signal conditioning can repair T3. The
[locked plan](plan.json) defines all candidates, exclusions, endpoints and gates.
The [diagnosis](DIAGNOSIS.md) uses only previously exposed data.

Two domains remain separate throughout:

- **New sessions:** archive ranks 10–12 for eight previously studied participants.
  Compare zero, 30 and 120 seconds of fresh labels on the same interval beginning
  at 122 seconds. History-based arms require 30 seconds of unlabeled normalization
  even when they consume zero fresh labels.
- **T3 continuation:** first two whole acquisition blocks beginning after 600
  observed seconds in each of eight known sessions, or the available subset.
  These are untouched outcomes in known recordings, not new T3 sessions.

History uses at most three strictly earlier days from the fixed development
index, with equal-day sufficient statistics. A prior-centered ridge fit adapts
that mapping to the permitted current prefix. Entire target participants are
excluded from global configuration selection; targeted T3 choices use only
strictly earlier T3 sessions, with a non-T3 fallback for the earliest session.
Insufficient fresh labels retain the historical prior, with explicit accounting.

## Reproduction

Use the existing Python environment with `.[dev,neural,benchmark]`; no native
Rust build is required. Acquisition and export instructions are in [DATA.md](DATA.md).
Prepared indices authenticate source members, feature arrays, targets, recording
boundaries and the committed study plan. Raw arrays and fitted models remain local.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m neural_experiments.v4.study select \
  --index data/neural_v4/development/index.json \
  --output results/neural_v4/source
python -m neural_experiments.v4.audit \
  --selection results/neural_v4/source/selection.json \
  --output results/neural_v4/source-audit.json
python -m neural_experiments.v4.public \
  --artifact results/neural_v4/source/selection.json \
  --audit results/neural_v4/source-audit.json \
  --output neural_experiments/results/v4-source-selection.json
```

Publish the source choices and passed audit before opening fresh outcomes. Then:

```bash
python -m neural_experiments.v4.study assess \
  --selection results/neural_v4/source/selection.json \
  --audit results/neural_v4/source-audit.json \
  --new-index data/neural_v4/new_sessions/index.json \
  --t3-index data/neural_v4/t3_blocks/index.json \
  --output results/neural_v4/assessment
python -m neural_experiments.v4.audit \
  --assessment results/neural_v4/assessment/assessment.json \
  --output results/neural_v4/assessment-audit.json
python -m neural_experiments.v4.public \
  --artifact results/neural_v4/assessment/assessment.json \
  --audit results/neural_v4/assessment-audit.json \
  --output neural_experiments/results/v4-study.json
```

Output directories cannot be reused. Source changes invalidate the assessment
gate. The independent auditor reconstructs feature recurrence, normalization,
fit equations, predictions, scoring masks, selections and aggregate results.
Synthetic tests cover temporal causality, future-label isolation, chronology,
fallbacks, export tampering and batch/stream parity.

Generate the published plots and reproduce the predetermined runtime case:

```bash
python -m neural_experiments.v4.figures \
  --study neural_experiments/results/v4-study.json --output neural_experiments/figures
python -m neural_experiments.v4.runtime \
  --model results/neural_v4/assessment/models/new_sessions-T2_day_187_decoding-selected_prior30.npz \
  --features data/neural_v4/new_sessions/prepared/T2/T2_day_187_decoding.features.npz \
  --plan neural_experiments/v4/plan.json --output results/neural_v4/runtime
python -m neural_experiments.v4.runtime_audit \
  --run results/neural_v4/runtime --output results/neural_v4/runtime-audit.json
python -m neural_experiments.v4.runtime_public \
  --run results/neural_v4/runtime --audit results/neural_v4/runtime-audit.json \
  --output neural_experiments/results/v4-runtime.json
```

The real benchmark uses three fast blocks and two60-second paced blocks after
other experiment compute has stopped. Timing replay does not expand scientific
scoring masks. Runtime measurements are host-specific, and the raw clock trace
stays local for independent auditing.

The primary accuracy endpoint is variance-weighted R² of a 20 ms cursor-to-target
direction proxy. It does not measure closed-loop task performance. Confidence
intervals resample eight people in the new-session domain and eight sessions
within the single person T3 in the continuation domain. Neither domain establishes
generalization to an untouched participant or a future prospective experiment.
