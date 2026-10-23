# V8: separate target, calibration coverage and representation failures

V8 tests a single preselected static decoder on new sessions while retaining a
small factorial experiment for interpretation. It changes neither the primary
instantaneous direction target nor the 120-second requested label budget. A
stable trial-onset direction is a separate diagnostic endpoint. Success on that
endpoint cannot establish recovery on the primary task.

The executable specification is [study-plan.json](study-plan.json). Its source
and selected configurations must be committed, independently audited and
published before fresh scoring. The preceding V7 scientific sources, failed
gates and receipts remain unchanged.

## What motivates the experiment

V7's fresh T2 frozen120 model had mean R² −0.0860 versus −0.0215 for zero
output. It incurred no calibration interruption inside the scored interval.
All three T2 sessions had negative R² under every learned arm. This is not
explained solely by the R² reference or by charging periodic interruptions.
However, the frozen model's angular errors were 80.13°, 80.47° and 79.60°,
compared with 90° assigned to zero output. Some directional alignment can coexist
with worse squared error. For unit targets, the identity
`MSE = 1 + mean(||prediction||²) - 2 mean(target·prediction)` distinguishes
output magnitude from alignment; residual bias and centered error provide
additional descriptive decompositions.

Independent prior audits reproduced data pairing, original targets and masks,
model fitting and full predictions. Their passage verifies implementation, not
biological target validity. Our instantaneous unit cursor-to-target vector is a
historical closed-loop intention proxy. Near-target normalization can amplify
small position changes. Long go periods can dominate row-weighted fitting,
while a short calibration interval may contain few distinct directions.

The pinned provider analysis uses a different endpoint: trial-onset windows,
300 ms mean position error, a near-target trial exclusion and nested
reaction-time selection. V8 does not reproduce that endpoint or copy its
acausal whole-block/session normalizers. It uses a fixed 40 ms reference and
0.2–1.0 second phase as a declared alternative. Neither target is independent
measurement of intended movement, and cursor velocity reflects the historical
controller. These limitations prevent a claim that one outcome establishes
true intention or absent neural information.

## Data and freshness

Development consists of the 176 already exposed prefixes: archive ranks 1–21
for the eight non-T3 participants and the original eight T3 sessions. Each is
capped at 600 recorded seconds. Unused tails and T3 continuation samples are
excluded. T2 ranks 1–15 choose the representation. Ranks 16–21 choose among the
eight frozen factorial cells. These are development partitions, not pristine
validation: earlier V2–V7 outcomes informed the hypothesis.

Fresh selection is fixed to regular archive ranks 22–24 for T2, T5, T6, T7,
T8, T9, T10 and T11. Authenticate the previous 21 members, retain shortages and
short recordings, and do not replace cases. Archive order is not chronological
day. New sessions are from known people; there is no untouched-participant
claim. No fresh behavior, targets, model scores or quality summaries may enter
representation selection or primary-arm selection.

All scientific times are recorded observation time, 20 ms per prepared bin.
Acquisition gaps do not supply elapsed real-world time. The first 30 seconds
fit feature normalization. Labels are purchased on [30,150), including invalid
go periods in the requested cost. Inference starts at 152 seconds and the
common primary score ends at the observed recording end or 600 seconds.

Calibration may use partial provider trials inside the purchased interval.
Every trial intersecting [30,150) is excluded from primary scoring, even if it
continues after 152. All arms share this exact exclusion and the original valid
go mask and 62-bin block burn. The final scored trial may be right-censored at
600; dropping it because it ends later would select on duration. Boundary
exclusion prevents directly fitting and evaluating the same trial but does not
make other trials or adjacent bins independent. Provider go intervals do not
identify task name, assistance, success or independent movement attempts.

## Locked model comparisons

The baseline is the V7 static representation: sqrt TX4.5 counts and log1p SBP,
a causal 200 ms EWMA, first-30-second mean/population-SD normalization, clipping
to ±8, mean-loss ridge 1 with a free intercept, and unit-disk output.

The representation search contains seven TX3+SBP options: single EWMA scales
50/200/500 ms crossed with ridge 1/10, plus combined 50 and 500 ms scales with
ridge 10. The original baseline is listed first as a fallback and wins exact ties. Each scale has its own
first-30-second normalizer; concatenated standardized scales are divided by the
square root of the number of scales. Blocks reset causal filter state. T2
ranks 1–15 rank these options by equal-session primary R² using the uniform,
all-go training configuration. Ties use the listed grid order.

The factorial is 2 × 2 × 2:

- Original versus source-selected representation.
- All purchased eligible go rows versus far, early-phase rows, fitting the
  **same instantaneous primary target**.
- Uniform row weights versus equal total weight per eligible provider trial.

The far-phase gate uses the position-error distance at raw trial onset plus
40 ms, strictly above the pinned participant-specific threshold, and endpoint
age `0.2 <= age < 1.0` seconds. No dynamic current-distance filter is added.
Phase training excludes trials beginning before second 30: their reference
behavior was not purchased. Equal-trial weights are computed only from
selected purchased rows. They reduce long-trial dominance; they do not balance
directional sectors or prove a task-coverage mechanism. Report actual row and
trial counts and effective sample sizes. Fewer than 50 fit rows yields zero
fallback, not a replacement candidate.

T2 ranks 16–21 choose one primary factorial cell by equal-session primary R²,
with the fixed axis order breaking ties. Retain all eight semantic cells even
if representation selection duplicates the baseline. Fresh scoring includes
all cells, but only this preselected cell can satisfy the recovery gate.

Controls are zero, the uniform purchased-label mean, one deterministic trial
label permutation using the selected configuration, and a separate stable
reference40-direction head. The permutation rotates purchased trial chunks by
one, mapping donor trajectories by normalized within-chunk index; it never
uses labels outside the purchased interval. Fewer than two trials produces an
explicit zero fallback. This is an alignment diagnostic, not a permutation
p-value. The stable-onset head uses far-phase training, selected representation
and weighting. It is scored against both unchanged primary direction and its
own far-phase target, without becoming an alternate primary after outcomes.

A current-target-plus-current-cursor control would define the primary target
algebraically. It is an oracle rather than evidence of neural decoding. The
constant control tests target prevalence; the trial permutation tests
neural-label alignment. Missing verified task-state metadata limits stronger
task-only controls and causal attribution.

## Descriptive mechanism evidence

Publish every factorial cell and paired phase, weighting and representation
contrasts within otherwise matching cells. Changes in training information and
interactions remain visible. A phase benefit alone cannot distinguish cleaner
labels from easier or different movement phases. An equal-trial benefit alone
cannot establish directional-coverage causality. A representation benefit
indicates predictive utility of that representation under this protocol, not
biological remapping or a specific neural mechanism.

For all exposed and fresh cases, retain calibration/test trial counts, octant
histograms, entropy and distribution differences, trial-fraction effective
sample size, far/phase proportions, instantaneous-versus-reference40 angular
disagreement, and TX3/TX4.5 activity summaries. These descriptive measures do
not select models. Keep neural sparsity, target variance, mean residual bias,
centered error, output magnitude and angular performance distinct. Trial-mean
secondary scores require at least two eligible trials; bin-pooled secondary
scores require at least 50 rows. They answer different questions and must not
be interchanged. Per-trial row records remain private; only aggregate count
distributions and effective sample sizes are published.

## Evaluation and decision

Every arm has the same primary mask. Compute float64 SSE/SST variance-weighted
R², axis scores, MSE and angular error, assigning zero output 90°. A recording
with fewer than 50 common scored rows remains unavailable. Pool bins within a
session for R²; average sessions within a participant, then give participants
equal weight. Costs retain all selected cases. Report all eight people and the
seven non-T2 people separately, including every material harm.

The primary T2 recovery gate requires all three requested sessions to be
scorable, mean R² at least 0.05, at least two positive session scores, and
positive paired mean gains over both baseline and zero whose descriptive
95% session-bootstrap lower bounds exceed zero. Use 10,000 draws and seed
20261023. Three sessions give limited uncertainty resolution. Cohort intervals
resample participants after averaging their sessions; bins are not independent
replicates. A missing case or alternate-target success cannot rescue a failed
primary gate. Even a T2 pass does not demonstrate cross-participant or clinical
efficacy; cohort harms remain part of the result.

## Independent verification

The auditor must bind source plans, two-stage selections, archive/member and
prepared-array identities, all model and trace artifacts, and all case
availability. It independently reconstructs target/trial geometry, purchased
masks, normalization, weighted normal equations, causal recurrence, output
projection, permutation mapping, full predictions, scores and registered gates.
Reference40 targets and context remain inaccessible to the feature-only
predictor. Perturbation tests must demonstrate that future labels/features and
labels outside the purchase interval cannot alter a fit or earlier output.

Retain failed attempts and exact executed source snapshots. Publish only
allowlisted aggregate results, identities and hashes; sample arrays, behavioral
positions, prediction traces and model coefficients remain local. Selection
and its passed audit are published before the single fresh assessment. No
fresh score-driven rerun, threshold revision or primary-arm replacement is
permitted.
