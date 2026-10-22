# V3 prospective session evaluation

This follow-up was designed after all V2 results and the random frozen-core control
were inspected. The nine participants and their first six selected sessions are
development data. V3 can test new session outcomes from these known participants;
it cannot establish generalization to previously unseen people. The target remains
the provider-derived unit cursor-to-target direction during valid go intervals,
an offline intention proxy rather than measured limb velocity or live control.

Commit `9ab460f` locked `plan.json` before new-session acquisition and assessment. The
bounded grid comprises threshold crossings alone or threshold crossings plus
log1p spike-band power, causal exponential-filter time constants of 0.1, 0.2, or
0.4 seconds, optional causal 30-second baseline subtraction, and mean-loss ridge
penalties of 0.1, 1, or 10: 36 candidate settings per label budget. The intercept
is unpenalized. Prefix-only population scaling uses floors of 0.1 for transformed
crossings and 0.01 for transformed spike-band power; standardized features are
clipped to ±8, and outputs are projected into the unit disk. Exact transform order,
state initialization, baseline predict/update order, and the meaning of the
mean-loss penalty must agree with the committed model implementation and tests.
No additional settings are implied. The original V2 sources, predictions, results,
and negative conclusions remain unchanged.

## What failed in the exposed V2 windows

The retained diagnostic inspected only the previously exposed first 600 seconds
of each original session and saved V2
predictions. It read no new-session outcomes and scored no retained unexposed tail.
All recomputed original scores matched their retained receipts. The table uses
the 120-second earliest-session calibration and equal-person means after equal
day/seed averaging on the five other original sessions per person.

| Method | R² | Mean angular error | Centered correlation | RMS prediction norm | Outputs outside unit disk |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original ridge | −4.041 | 86.72° | 0.048 | 1.742 | 58.0% |
| Scratch temporal model | −3.108 | 86.90° | 0.067 | 1.613 | 64.5% |
| Learned frozen core | −1.591 | 87.24° | 0.058 | 1.137 | 44.0% |
| Fine-tuned learned core | −3.546 | 88.50° | 0.056 | 1.720 | 70.4% |
| Random frozen core | −0.562 | 86.12° | 0.060 | 0.714 | 17.2% |

Target unit-vector norms agree to within 6 × 10⁻⁸. A zero-vector prediction scores
−0.0176 on the same future-day windows. Removing the learned frozen model's mean
residual bias using evaluator labels would still leave mean R² at −0.408. Projecting
its outputs into the unit disk gives −0.621; even an evaluator-fitted nonnegative
scalar per trace reaches only −0.0018. These last quantities are exposed-data
diagnostics, not deployable models or fresh validation results. They show that
amplitude and offset errors are substantial, while useful cross-day directional
information is weak. Output clipping alone does not explain a useful decoder.

Calibration distribution and recording drift also matter. In the first
120-second prefixes, the mean target-direction norm is 0.504 for T2 and 0.537 for
T7, showing directional imbalance. T3 and T6 have 97.9% and 89.6% of channels below
the original square-root-count standard-deviation floor. On later exposed
sessions, the RMS standardized channel-mean shift reaches 1.24 for T8 and 1.50 for
T11; their absolute standardized feature 99th percentiles are about 10. These are
observations, not proof of a single causal mechanism. Electrode identity/layout
consistency and raw target/mask reconstruction are checked separately by the data
audit.

## Prospective data boundary

1. Commit the acquisition rule before acquiring new outcomes: select the seventh,
   eighth, and ninth eligible complete decoding-session members in each pinned
   Dryad participant archive, using archive order and the same fixed eligibility
   rule as V2. Retain all acquisition failures, short sessions, and exclusions.
   Archive order is not chronological order. Numerical post-implant days must be
   reported after selection; use “new session” unless a specific forward-day
   relation is verified.
2. Preserve the pinned provider version, archive identities, streamed-member
   receipts, complete member hashes, preparation source, and prepared feature and
   target hashes. A published full-archive digest does not imply that a locally
   streamed prefix verified that entire archive.
3. Select every model setting using the already-exposed original first-six-session
   data only. Commit the complete plan and freeze all nine outer-fold selections,
   source input identities, and their validation scores before the study runner
   opens any fresh assessment label file, including calibration labels. Fixed data
   preparation and metadata inspection do not authorize scoring.
4. Evaluate the fixed collection once. Keep every per-session prediction and all
   failed/unavailable cases. No new setting, label budget, mask, time window, seed,
   participant subset, or model choice may be selected using fresh results.

Existing `[600,1200)` tails would provide only eight eligible participants; T6 has
no retained samples beyond 600 seconds. They are not substitutes silently added to
the new-session endpoint. The original nine people and 54 session identities have
already influenced development, and no reused window may be described as an
untouched participant or session replication.

## Source-only selection and calibration

For each outer participant, exclude that entire participant from source training,
source validation, hyperparameter selection, and any learned shared component.
Evaluate the locked candidate grid on the other eight participants' original
sessions: fit each source session's declared calibration prefix, then validate on
its already-exposed `[302,600)` interval for both budgets, clipped to its actual
duration. Compute
equal-session means within each source participant, then equal-participant means.
Report source participants with no eligible validation interval and the exact
selected setting; deterministic ties and all candidate settings belong in the
plan. Do not use any outer participant's development scores to choose that outer
fold's setting. Source-only selection does not erase prior architecture-design
exposure to this known cohort.

Select the best threshold-crossing setting, the best combined crossing/power
setting, and the overall best setting using this source-only rule. The primary
new-session arm is the overall-selected pipeline with 120-second daily
calibration. The family-specific arms are declared ablations, with no replacement
of the primary arm after scoring. Compare against the original V2 fixed ridge
refitted to the same daily budget, plus zero-vector and prefix-mean controls.

The primary task is **daily supervised calibration**, not V2's one-earliest-day
mapping. Each selected new session provides labels only from its first 120 seconds;
the primary scoring interval is `[122,min(600,duration))`, with a two-second
calibration-to-evaluation gap. Time is observed recording time, never compressed
movement-mask time. Record both available prefix labels and actual training rows,
including the common first-62-bin burn-in at each recording block. The prefix-mean
control uses the same effective labeled rows. Fit normalization and supervised
parameters only from the declared prefix. Causal filters incorporate the current
observation before its prediction, without masks or labels entering filter state.

The 300-second secondary calibration uses `[302,min(600,duration))`. To estimate
the effect of more calibration, score **both 120- and 300-second models on the
same `[302,min(600,duration))` interval**. Report this matched comparison separately
from the primary 120-second task. Secondary-prefix labels must never enter a
120-second fit or trigger a primary-model change.

Keep original recording gaps, block resets, and non-go observations in context.
Never concatenate disjoint go intervals or normalize with future-session/block
statistics. All methods use the same original 20-ms target and go mask; no
outcome-driven near-target exclusions or target redefinition. Require at least
50 scored evaluation bins by a fixed rule, retain null results and exact reasons,
and never replace an inconvenient session. Preserve the original earliest-day
frozen mapping as a separate between-session stress arm; identify chronological
direction rather than assuming every newly acquired session is later.

## Metrics and useful-accuracy decision

Report independent pooled-SSE R² and per-output R², angular error, zero-output
counts, calibration counts, per-session/day/person scores, and absolute performance
of zero-vector and prefix-mean baselines. Aggregate equal sessions within person,
then equal people, averaging seeds before person-level comparisons. Use paired
person-level bootstrap uncertainty with the resampling rule fixed before scoring;
report the small cohort and source-fold dependence rather than treating time bins
or sessions as independent people. Do not let relative improvement over a poor
neural baseline substitute for positive absolute accuracy.

The locked operational gate requires all of the following:

- All nine enrolled participants eligible for the primary matched evaluation.
- Equal-person mean R² at least 0.10, with its person-bootstrap 95% lower bound
  above zero.
- Mean R² improvement at least 0.05 over both zero-vector and prefix-mean baselines.
- Positive person-level R² for at least seven of nine enrolled participants.
  Missing participants remain visible and fail the nine-person eligibility gate;
  they cannot silently reduce the success threshold.
- Mean angular error at most 80°, with the corresponding baseline comparison
  reported. The 90° directionless reference is descriptive, not a substitute for
  a data-matched baseline or randomized negative control.

These are declared engineering/analysis thresholds for this offline proxy; they
are not clinical utility standards. Report per-person harms and all failed gates.
Any useful-accuracy conclusion applies only to the locked daily-calibrated task
and these newly scored sessions of known participants. Genuine new-person
generalization still requires a cohort not used in model or protocol development.

## Independent verification contract

The auditor must bind the exact acquisition/preparation/input/plan/model/prediction
artifacts and check that every source-fold participant set excludes its target.
It must reconstruct prefix bounds and effective training masks, verify that fresh
scoring intervals and masks match across methods, retain all unavailable cases,
and independently recompute all scores and person-level summaries from saved
predictions. Predictor calls receive only observations and declared recording
boundaries. Tests must attack later-label leakage, future-observation dependence,
calibration-budget mismatch, wrong-session/mask acceptance, frozen model identity,
and tampered model/prediction receipts. `audit.py --selection` checks source-grid
traces, whole-person-excluded rankings, and source/index/plan identities without
receiving an assessment index. Its passed receipt gates assessment-label access.
`audit.py --run` additionally verifies the complete case inventory, calibration
normalizers, ridge normal-equation residuals, full-session predictions using an
independent NumPy recurrence, and all evaluation summaries. Neither mode refits
source-grid models. The source-only gate passed on all 3,528 candidate cases, 54
outer choices, and 10 unavailable cases before fresh assessment. This document
does not itself claim a completed assessment audit or fresh accuracy result.
