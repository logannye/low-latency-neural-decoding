# V2: accuracy, calibration efficiency, and participant transfer

The original H1 study is frozen. V2 code lives here rather than changing its
accepted decoder. The H1 extension is development research on already exposed
participants/days; it cannot create another untouched H1 test set.

The new scientific question is whether a compact temporal core trained on other
people improves a new person's decoding with the same small calibration budget.
The machine-readable [transfer plan](transfer_plan.json) specifies the primary
contrast before model fitting or evaluation. A run binds its exact plan, source,
prepared index, model, and prediction hashes. Data acquisition and synthetic
correctness tests precede any real-data performance claim.

## Data and estimand

The initial provider is BrainGate's [2026 array-performance release](https://doi.org/10.5061/dryad.x0k6djj1h),
pinned to Dryad version 462702. Nine people have decoding recordings; the fourteen
people in the array-yield dataset are not fourteen decoding participants. Select
up to six complete session members from each participant archive in archive
order, without inspecting model outcomes, then order by numeric implant day.
This deterministic convenience sample is not representative sampling across the
full 729 sessions or the population of BCI users. Report archive/member selection
and missing data explicitly. No participant is discarded because decoding is poor.

Before any transfer training or test evaluation, the source optimization budget
was amended from 600 to 2,400 steps, with source-only validation checkpoints at
600, 1,200, and 2,400. At the session caps, 600 batches would provide only about
one pass over eight people's source chunks. This amendment increases coverage;
it was not triggered by a target-participant result. The original plan commit and
amendment remain in Git history.

Combine two adjacent 10 ms count bins within the same recording block into one
20 ms observation. The paired target uses the second bin's contemporaneous
cursor and target position. Never average future targets into an earlier output.
The endpoint is the unit vector from cursor to target during valid movement-attempt
intervals. It approximates intention; recorded cursor motion was produced by an
existing controller and is not independent ground-truth limb velocity. Keep
non-go observations in sequence history, and keep scorer targets/masks out of
inference. Reset only at recording block boundaries. Do not join across gaps.

The release's raw-voltage filtering phase, buffering, and threshold-estimation
window have not been established. Accordingly, this study measures causal
decoding from supplied features. It does not establish neural-acquisition-to-output
latency. Do not reuse published whole-block centering or whole-session scaling.

## Comparison and leakage boundaries

Every outer fold excludes one entire person from source training and validation.
Other people's last selected session validates the source checkpoint, and their
earlier sessions train it. Fixed step counts, model family, preprocessing,
calibration budgets, and primary contrast are chosen without target test scores.
Pretraining is performed within each fold; no external pretrained weights or
potentially overlapping H1 participant enter this cohort's source pool.

The input adapter is participant-specific because electrode indices do not imply
corresponding neurons. Its output enters a shared causal width-32 residual TCN;
identity paths preserve a direct gradient and current-feature path through the
five temporal layers. Each new
person receives a fresh adapter and output head. Compare a frozen pretrained core,
full fine-tuning, the identical architecture trained from scratch, and ridge.
Every arm uses precisely the same 30/120/300-second prefix and eligible label rows.
Target input statistics also use only that prefix. Count labeled seconds and
adaptation computation separately. This is supervised few-shot transfer, not
zero-shot decoding.

Score the same first-session suffix after 300 seconds for every feasible budget;
score subsequent days separately with no updates. Report unavailable arms instead
of silently extending budgets or replacing short recordings. Source/evaluation
session time caps are fixed in the plan and include unscored time. Training chunks
carry only real preceding observations from their own block and never score padding.

## Evidence and engineering gates

The primary contrast is frozen-core pretraining versus scratch with 120 seconds
of calibration on future days. Equal-weight people and equal-weight days, rather
than letting long sessions dominate. Report all three seeds, both coordinates,
angular error, all people, all days, and the absolute paired R² difference. Bootstrap
people and perform an exact participant-level sign-flip analysis; neither seeds
nor thousands of time bins are additional independent people. Other comparisons
remain descriptive, including whichever method has the best numerical score.

Synthetic tests check future-input independence, block reset isolation, training
and normalization prefix boundaries, incremental versus whole-sequence parity,
and rejection of changed artifacts. Independently reconstruct scores from saved
predictions with sklearn. Test traces and model weights remain in ignored local
storage; publish aggregate evidence and a reproducible acquisition/training path.

Measure the exact prespecified model's complete incremental step, including
transforms and state updates, after training has stopped. Keep service time,
scheduling delay, release-to-completion, deadline misses, initialization, and
numeric state separate. Repeat fast replay and paced 20 ms releases. A p99 below
2 ms is a provisional engineering target, not a hard real-time guarantee. A null
transfer result is retained and limits the claim; it does not trigger test-set tuning.
