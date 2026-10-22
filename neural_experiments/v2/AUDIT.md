# Independent V2 protocol audit

This audit concerns the nine-person BrainGate transfer study. It is separate from
the V1 H1 study and from the V2 H1 accuracy experiment. The evaluator may read
targets and movement masks; a replayed predictor receives neural observations and
recording-boundary information only.

## Required evidence

- Pin Dryad version 462702 and all nine participant identifiers. Preserve the
  archive-order acquisition rule, selected complete member names, skipped or
  unavailable files, and per-member byte hashes. A published full-archive digest
  is not a verified local archive digest when only a prefix was streamed.
- Preserve all selected sessions in the prepared index. Sort their numerical
  post-implant days only after selection. Exclusions must have an explicit
  outcome-independent reason and must remain visible in the final denominator.
- The held-out person must be absent from source training, source validation,
  hyperparameter selection, and checkpoint selection. Source-session lists must
  resolve back to the hashed index, not merely to a declared participant label.
- For each held-out person, all methods receive the same earliest-session prefix
  at each 30, 120, or 300 second budget. A budget is elapsed feature-ready time;
  movement masks select eligible labels inside that prefix and do not compress
  the time axis. Save both elapsed bins and eligible labeled-bin counts.
- Use an identical same-session suffix beginning at 300 seconds for every budget
  and method. Report later-session evaluation separately. A short first session
  must have a visible, predeclared disposition; do not silently substitute a
  better-scoring session.
- Fit target-specific preprocessing only on the permitted prefix. Target-session
  future statistics, full-block centering, and full-session scaling are forbidden.
  Save the source or calibration ranges used for each fitted transform.
- Preserve block and session boundaries. Filtering, convolutional history, and
  recurrent state reset at acquisition discontinuities, but not at movement-mask
  or trial transitions. Never stitch disjoint movement intervals into a sequence.
- Predictions must cover all requested bins. The scorer applies the same hashed
  mask to every method. Missing, duplicated, nonfinite, or truncated predictions
  fail the audit instead of silently reducing the scoring set.
- Bind configuration, source snapshots, models, predictions, prepared arrays, and
  reported summaries to exact hashes. Preserve failed runs and issue fresh audit
  receipts rather than overwriting prior evidence.

## Numerical checks

The audit reconstructs pooled-SSE and per-output R² with scikit-learn
from saved predictions and independently loaded labels. It reconstructs
per-day scores, equal-person aggregates, angular-error counts, and paired
person-level uncertainty without importing the experiment scorer or predictor.
Sessions and trials are repeated measurements within a person, not independent
people. Nine held-out people remain a small sample.

Tests attack the protocol boundaries: later-label and future-observation
perturbations, unequal calibration budgets, changed or missing artifacts,
participant overlap, mismatched scoring rows, and recording discontinuities.
Numerical agreement alone cannot prove that the claimed training data were the
only data consumed; source inspection and execution-boundary tests are also
required.

## Claim boundary

Refitting one fixed recipe independently per person tests recipe portability.
Comparing a source-trained core against the same architecture trained from scratch
on an identical target-label budget tests calibrated transfer to a held-out person.
It is not zero-shot transfer. Electrode indices do not provide neuron
correspondence across people, and a target-trained input adapter or output head is
part of calibration. Offline target-direction decoding does not establish
closed-loop cursor performance. Supplied preprocessed features support a
feature-ready timing claim unless their acquisition processing is separately
validated for causal availability.

## Execution record

`audit.py` imports no experiment scorer, model, importer, or predictor. Its CLI
requires a complete run and writes a fresh JSON receipt:

```sh
PYTHONPATH=src .venv/bin/python -m neural_experiments.v2.audit \
  --run results/neural-v2/TRANSFER_RUN \
  --index data/neural_v2/INDEX.json \
  --output results/neural-v2/TRANSFER_RUN/independent-audit.json
```

Use the actual retained paths; the names above are placeholders. Run and model
artifact paths are repository-relative; prepared-array and raw-member paths are
relative to the input index directory. The audit validates the saved consumed
source snapshots, so later edits to unrelated analysis or documentation files do
not invalidate the original execution identity. Its receipt binds its own source,
the complete run manifest, configuration, input index, and reconstructed summary.

The source schedule is the pre-execution amendment at `2154dae`: 2,400 source
updates and source-only checkpoint selection at 600, 1,200, and 2,400 updates.
The original `c0a7480` plan remains in Git history. This amendment preceded transfer
fitting and held-out scoring; architecture, target-label budgets, and the primary
comparison were unchanged. No data-dependent amendment is permitted by this audit.

The auditor checks all 270 registered target cases: deterministic ridge at seed 0
and three neural methods at seeds 0, 1, and 2, each at three budgets for nine people.
Every absent case must have the exact prespecified duration or training-label
eligibility reason. It independently recomputes the first-62-bins-per-block loss
exclusion for all methods and both source and target normalizers. Frozen-core
exports must exactly equal their held-out-person source core. Future-day metrics
use all eligible bins, with undefined R² preserved as null; same-session suffixes
with fewer than 50 eligible bins retain full predictions and a null-metric reason.

The focused suite contains 24 tests. A full synthetic receipt fixture covers nine
people, 27 source cores, 90 feasible 30-second target models, 180 unavailable longer
budgets, and 180 saved traces. It compares the runner's summary against independent
aggregation and rejects a post-receipt prediction edit. These synthetic models are
untrained fixtures and provide no evidence of biological prediction accuracy.

A separate smoke test runs the production training path with explicitly reduced
synthetic settings and three artificial participants. It pretrains three outer
folds, fits all four methods, restores 12 exported models, checks 24 prediction
traces and 12 unavailable cases, and rejects overwrite. A gradient-step test verifies
that the shared core appears once in the optimizer while inactive participant
adapters remain unchanged. An empty future-session regression exposed sklearn
ridge's zero-row rejection; the runner now retains an empty prediction array and
null metrics for that case. These checks do not change the locked study plan.

No completed real nine-person run has yet passed this audit. A real-data execution
receipt is required before reporting audited cohort results. Hash consistency and
numerical agreement do not prove exclusive training-data use by themselves; the
source review and perturbation tests support that separate claim. This audit does
not independently reproduce optimizer updates or validate upstream acquisition
filter causality, runtime deadlines, or closed-loop clinical performance.
