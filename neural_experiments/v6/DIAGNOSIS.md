# V6 exposed-data diagnosis: more labels and exact-model confirmation

This is **post-hoc exploratory evidence**, informed by the observed V5 failures.
The pilot uses only the already exposed first 600 seconds of all 15 T2 sessions
through V5. It reads no raw tails, T3 continuations or fresh V6 outcomes. The
[public aggregate](../results/v6-diagnosis.json) retains every candidate fit's
metrics, all failed decisions, exact input/source hashes and model identities.
Numeric models and sample arrays remain private.

## What was tested

The pilot retained 1,440 fits and 15 zero-label historical-prior exports. Each
training budget of 60, 120, 240 or 360 seconds begins after a 30-second unlabeled
warmup. Twelve candidates cross two feature families, three heads and two
normalizers: TX4.5 with 0.4-second smoothing or TX4.5+SBP with 0.2-second smoothing;
prior-centered gamma 10, ridge 1 with a free intercept, or ridge 10 with exactly
zero intercept; first-30-second or all-features-available-at-fit normalization.

Each candidate is first fitted on the first half of the prospective training
interval. Its inner validation starts two seconds later and ends at the full
training stop. The lowest coordinate-MSE candidate is chosen there, before any
confirmation label is used. That candidate is refitted on the full training
interval, exported and reloaded. Confirmation evaluates this exact frozen
artifact over the next 60 elapsed seconds, excluding the first two seconds.
There is **no refit or normalizer update after confirmation**. A separate fixed
combined-feature ridge-1 candidate provides a direct comparison with this
12-way inner selector.

Every candidate is scored on the same **452–600 second** suffix. Confirmation
ends at 450 seconds for the longest stage, leaving a further two-second gap.
This endpoint differs from V5's primary 272–600-second endpoint. A 360-second
training stage plus confirmation requests 420 seconds of labels and reaches a
decision at 450 seconds after recording begins. Historical prior0 instead fits
no query labels and confirms on 30–90 seconds, costing 60 labeled seconds.
Historical fits use the last three strictly earlier same-layout exposed sessions.

Confirmation reports pooled and separate first/last-half errors, directional
coverage, gains over zero and the training-label-mean predictor, and R² using
each window's own target mean and variance. A half with narrow directional
coverage can have strongly negative R² despite modest improvement over zero;
those are different comparisons. Confirmation and inner validation apply the
62-bin acquisition-block burn; the final common scoring suffix uses all valid
scored bins. The retained V5 artifact comparison also reports burned windows,
alongside separate exact original-endpoint metrics.

## Main findings

The fixed combined-feature, free-intercept ridge-1 fit with normalization through
the training stop benefits from longer calibration on the latest three exposed
sessions:

| T2 day | 120 training seconds | 240 training seconds | 360 training seconds |
| --- | ---: | ---: | ---: |
| 214 | −0.0731 | 0.1054 | 0.1007 |
| 228 | 0.0389 | 0.0060 | 0.0912 |
| 236 | −0.0132 | 0.0552 | 0.0830 |
| Mean | −0.0158 | 0.0555 | 0.0917 |

Across all 15 exposed T2 sessions, the corresponding means are 0.0986, 0.1529
and 0.2032. The older nine have substantially better results than the later six;
these pooled improvements do not establish reliable recovery under drift.
Normalizer differences are usually smaller than the effect of additional labels,
with a notable exception: day 209's 120-second fit improves by about 0.162 R²
when using all features available at fitting rather than only the warmup.

The inner selector can discard useful later fits. At 360 seconds on days 214
and 236 it selects a strongly regularized zero-intercept head, scoring −0.0377
and 0.0369 instead of the fixed ridge-1 candidate's 0.1007 and 0.0830. The final
production hypothesis therefore fixes the strong candidate rather than choosing
among 12 models from one short inner interval. All rejected alternatives remain
in the aggregate.

Confirming the exact artifact addresses V5's validation-versus-refit mismatch,
but **confirmation alone does not guarantee later accuracy**. The day 214
120-second combined fit has next-window coordinate MSE about 0.378 versus 0.500
for zero, yet its final R² is −0.0731. Its first confirmation half loses to the
training-mean baseline, whereas the second half wins. The original V5 full
refit similarly helps on the next independent window but acquires a larger
later bias than its half-prefix fit. An immediate validation improvement cannot
be treated as stable performance over the rest of the session.

**Day 187 is a concrete counterexample to a broad reliability claim.** Its fixed
120/240/360-second models all look useful during confirmation but have final
R² −0.3421/−0.5443/−0.3610. Even delayed, disjoint confirmation cannot identify
arbitrary later changes in the decoding relationship.

## Strictly earlier evidence and the bounded next hypothesis

For a hypothetical day-214 query, the prior three exposed sessions are days
187, 194 and 209. Their fixed-model mean final R² at 120/240/360 seconds is
−0.1395/−0.1364/−0.0916: none establishes useful decoding. For day 228, earlier
days 194/209/214 give −0.0498/0.0802/0.0623; for day 236, earlier days
209/214/228 give −0.0171/0.0504/0.0629. Thus earlier outcomes can inform a
calibration budget, but they are evidence about risk and relative benefit, not
a promise of positive query-session accuracy.

A post-hoc illustration combining a fixed strong candidate, a conservative
budget floor and confirmation can use 360/240/360 training seconds on the latest
three sessions, producing mean final R² about 0.0632. This illustration is
**not fresh validation**. The 12-way selector with strict two-half confirmation
instead abstains on all three, at 420 requested label seconds each. Both false
acceptance and rejection of potentially useful decoders must remain visible.

V6's proposed finite policy grid fixes the combined ridge-1 model, varies two
confirmation margins and two pooled-R² thresholds, and uses strictly earlier
learning curves for its budget floor. Threshold selection excludes the entire
target participant. Fresh assessment must determine whether this improves
accuracy and acceptance enough to justify its extra label and waiting costs.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src:. \
  .venv/bin/python -m neural_experiments.v6.diagnosis \
  --output results/neural_v6/diagnosis-reproduction
PYTHONPATH=src:. .venv/bin/python -m neural_experiments.v6.diagnosis_summary \
  --pilot results/neural_v6/diagnosis-reproduction/diagnosis.json \
  --output results/neural_v6/diagnosis-reproduction/summary.json
PYTHONPATH=src:. .venv/bin/python -m neural_experiments.v6.diagnosis_public \
  --run results/neural_v6/diagnosis-reproduction \
  --output results/neural_v6/diagnosis-reproduction/public.json
```

The first run took 20.05 seconds with one-thread caps. The scripts authenticate
both exposed indexes before opening arrays, preserve a preexecution plan and
source snapshots, and refuse existing output folders. Pilot receipt SHA256:
`040d7d4db80fb24b35d8a0308aa68d10fdbfbc072bdaf7d4cf6cf02bb81ff833`.
The generic fitting helper has focused tests for prefix-only label access,
normalization boundaries, arbitrary inner stops, exact ridge scaling,
serialization and rejection of incomplete requested intervals. V2–V5 source and
results were not modified.
