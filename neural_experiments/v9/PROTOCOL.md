# V9: trial-balanced calibration and task-defined targets

This protocol is fixed before model development and fresh signal conversion. It tests whether calibration that accounts for the task's observed directional support, together with a provider-informed trial target, rescues T2. The machine-readable [study plan](study-plan.json) fixes every selection, model comparison, score and decision. The [data contract](DATA.md) identifies exactly what the public recordings do and do not establish.

## Why this experiment

V8 found only 7–10 distinct T2 calibration trials in two recorded minutes. Equal-trial weights did not add experience, and its early/far subset left very few training observations. The new feasibility analysis uses only already exposed data. T2's go directions are largely cardinal: an eight-sector quota would incorrectly demand directions the task often does not supply. Four 90-degree sectors centered on the cardinal directions give a meaningful, fixed coverage measure.

The provider's trial endpoint also differs from V8's fixed early direction. It normalizes the mean cursor-to-target displacement over a 300 ms window after a reaction offset. The onset-plus-40-ms reference determines far-distance eligibility. V9 validates this distinction against the pinned implementation, while keeping inference causal and requiring complete windows inside the observed go interval. This is a provider-informed geometric proxy, not independently measured intention or an exact reproduction of provider dSNR.

## Cohorts and access order

Development reuses all 200 previously exposed V8 prefixes without extending them. Model development uses the 24 sessions each from T2, T7, T9 and T10. Only T2 archive ranks 16–24 select reaction offset and regularization. These nine exposed sessions include the previous fresh T2 assessment; they are development now, not independent confirmation. Other source results are retained but do not select the recipe.

Fresh acquisition is fixed at archive ranks 25–30 for the same four participants: six T2 sessions and 18 comparison sessions. Selection does not inspect duration, task coverage or outcomes. All selected cases, shortages and failures remain in the record. There are no replacements. Archive rank is not necessarily chronological post-implant day. Each prepared prefix ends at 600 recorded seconds or recording end, whichever comes first. Fresh sessions train neither each other nor the source choices.

The acquisition plan is committed before raw downloads. The scientific plan and producers are committed before source fits. An independent numerical implementation must reconstruct source fits, predictions and choice before that choice is committed. Fresh signal conversion and one-shot scoring occur only after this seal. Raw records and sample-level outputs stay local and ignored; public artifacts contain aggregate evidence and immutable hashes.

## Calibration as an observable policy

Normalization uses the first 30 recorded seconds. The original startup control purchases seconds 30–150. The maximum-budget control purchases seconds 30–390, a 360-second budget. The new policy recruits complete eligible go intervals in observed completion order and stops when it has **at least 24 trials and at least four in each of four cardinal-centered sectors**, or reaches second 390.

A sector is `floor((atan2(y,x)+π/4)/(π/2)) mod 4`, using the trial's fixed 40 ms reference direction. Sector assignment does not depend on the chosen reaction offset, training target or model outcome. No future directions are inspected to skip or replace trials. Equal-sector weights give each occupied sector equal total weight, then divide that weight equally across its trials. Missing sectors are reported; weighting cannot manufacture them.

The policy recognizes completion only at `(trial_stop_bin + 1) × 20 ms`, after observing that conservative boundary. The paired end can precede the original raw go end by 10 ms. Right-censored trials cannot satisfy the quota. A timeout fits the available eligible trials rather than excluding the session. Continuous fits require at least 50 selected bins; trial fits require at least one eligible vector. Below those minima, use zero output. A single trial can only fit a constant through the free intercept and is not evidence of directional coverage. A short recording cannot deploy before its requested policy stop. Report requested and observed calibration time, completed trial and sector counts, deficits, quota attainment and fallback status for every session. These are recorded-time costs, not wall-clock estimates through acquisition gaps.

The exposed feasibility check found the final eligibility quota attainable by second 390 in 20/24 T2 sessions, including 6/9 of the selection sessions. Extending to second 450 added no attainable T2 cases after the geometry and burn-in rules. All 24 source sessions from each of T7, T9 and T10 attain it by 390. The 390-second deadline therefore preserves more held-out observations without changing this source feasibility result. These counts describe the exposed source only and make no fresh-performance claim.

## Validated window geometry

The two reaction offsets are 0.2 and 0.8 seconds; every selected window contains 15 complete 20 ms pairs. The anchor is the first pair whose second-sample go age minus 10 ms reaches the requested offset, compared in rounded integer centiseconds to avoid floating-boundary ambiguity. The resulting physical start is the requested offset or 10 ms later. All feature and target windows use those same pairs. Behavioral positions are the retained second 10 ms sample of each pair; V9 does not claim exact equivalence to a raw 10 ms provider calculation.

Eligibility is common to both offsets: the full span from the early window's start through the late window's end must remain in one go interval and one recording block, pass the 62-bin block burn-in, and have finite nonzero displacement. The target may move by at most 1% of the 40 ms reference distance over that span. Directional coherence, the norm of the mean unit displacement, must be at least 0.95. The mean displacement in each candidate window must be nonzero. The reference distance must exceed the provider's participant-specific far threshold. Training additionally requires a go onset at or after second 30, a complete observed go, and completion within the purchased interval.

These rules test the internal consistency of a target-directed movement window. They do not identify task name, assistance, success, actual intended movement or electrode physiology. Eligible counts, sector deficits and unavailable outcomes are retained. The unchanged primary endpoint does not apply these geometric filters.

## Models and source choice

Continuous controls retain the V8 decoder: TX4.5 and spike-band power, square-root/log transforms, causal 200 ms exponential filtering, ridge 1, first-30-second normalization and unit-disk projection. This preserves a meaningful original-baseline comparison.

Trial readouts use the same neural channels and transforms with a causal trailing 300 ms boxcar, reset at block boundaries. A training trial supplies one feature vector at the window's last bin. Two matched targets are tested: the original instantaneous direction at that bin, and the normalized mean displacement over the same window. Uniform trial weights and equal-sector-then-trial weights are compared on exactly the same selected trial inventory. At runtime the readout uses only neural features and block boundaries; it does not receive trial phase, cursor, target or task labels.

The fixed primary is `quota_sector_mean`. Four source recipes vary only reaction offset and ridge: (0.2, 0.1), (0.2, 1), (0.8, 0.1), (0.8, 1), in that order. Select the largest equal-session mean primary R² over exposed T2 ranks 16–24; ties choose the first exact maximum. Every recipe uses the same primary mask and eligible trial inventory. A source failure does not authorize replacing the recipe after fresh testing.

Twelve fixed assessment arms separate useful causal contrasts:

| Arm | Purpose |
|---|---|
| `baseline` | Original continuous readout, startup 120 seconds |
| `duration` | Same readout, full 360-second budget |
| `quota_continuous` | Same readout, exactly the quota policy's stop |
| `trial_startup_sector_mean` | Selected trial recipe, startup budget |
| `trial_duration_sector_mean` | Selected trial recipe, full budget |
| `quota_uniform_endpoint` | Equal-trial weights and endpoint target |
| `quota_sector_endpoint` | Equal-sector weights and endpoint target |
| `quota_uniform_mean` | Equal-trial weights and mean-displacement target |
| `quota_sector_mean` | Fixed primary: quota, sector weights, mean-displacement target |
| `zero` | Zero-output control |
| `constant` | Mean instantaneous label from the full purchased interval |
| `trialshuffle` | One fixed cyclic rotation of primary training-trial targets |

The single trial-label rotation is a negative control, not a permutation null distribution. No privileged behavior-only predictor is counted as a neural decoder. Exported models are reloaded before scoring; all private fitted artifacts and prediction traces are preserved for reconstruction.

## Common held-out evaluation

All arms use the identical primary suffix, seconds 392–600, restricted to valid go bins after block burn-in. **Every go interval intersecting the maximum purchased interval [30,390) is purged from every arm**, even if an arm stopped labeling much earlier. Thus training and evaluation never share a go interval, and a shorter-label method cannot gain an easier test set. Short recordings retain unavailable scores. The primary R² requires at least 50 eligible bins and finite, nonzero total target variance.

The primary remains continuous instantaneous cursor-to-target direction. Summed squared error and target variation are computed in float64, producing variance-weighted two-output R². Angular error and summed MSE are secondary descriptions. The trial diagnostic evaluates each method's output at the selected window end against both the mean-displacement target and the endpoint target, using a common eligible held-out trial inventory. Each secondary go must be complete and noncensored, with its entire common 45-bin span inside the primary mask; the continuous primary retains eligible observations from a right-censored final go. It requires at least two trials. Success on that changed endpoint cannot satisfy the continuous recovery decision.

## Decisions and uncertainty

Useful T2 recovery requires all six requested T2 sessions to be scorable, mean primary R² at least 0.05, at least four positive sessions, and positive paired gains over both the original startup baseline and zero. Each paired mean and the lower bound of its session-bootstrap interval must be strictly positive. A separate incremental-benefit gate additionally requires a positive paired mean and lower bound over the unchanged full-duration decoder. If only longer calibration works, report that mechanism rather than crediting trial balancing.

Target and weighting contrasts use the exact same quota stop, trials, representation and regularization. They are descriptive, as are the budget comparisons and participant differences. No multiplicity-adjusted causal or significance claim follows from them. Average sessions within each person, then people equally; resample sessions for T2 descriptions and participants for cohort descriptions, using 10,000 replicates and seed 20261023. Six sessions from one person are not six independent participant replications.

Quota attainment is reported separately and never determines score inclusion. All primary failures, negative transfers, fallback cases and unavailable scores remain public. Fresh comparisons cannot change the chosen primary, and unsuccessful finite fixes cannot prove T2 is undecodable. Offline replay does not establish closed-loop performance, adaptation by the user, clinical benefit or a hard real-time guarantee.
