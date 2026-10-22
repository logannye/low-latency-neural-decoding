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
  --run results/neural-v2/transfer-v1 \
  --index data/neural_v2/index.json \
  --output results/neural-v2/transfer-audit-v1.json
```

The command above produced the retained real-study receipt; use a fresh output
name when repeating the audit. Run and model artifact paths are repository-relative;
prepared-array and raw-member paths are
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

The completed `transfer-v1` run passed independent audit: 260 target models,
1,560 saved traces, 27 source cores, all 54 selected complete session members, and
10 explicitly unavailable target cases. The unavailable cases are T6's 300-second
budget for all registered methods/seeds; its first session contains 224.46 seconds.
T6 remains in the 30- and 120-second future-day comparisons, and its unavailable
same-session suffix remains visible. No audit failure or outcome-dependent recipe
change occurred. The aggregate publication retains all per-session metrics and
calibration counts.

At the registered 120-second future-day comparison, the frozen pretrained core
minus scratch mean R² difference is **+1.517661**, with a person-bootstrap 95%
interval of **[+0.494858, +2.714568]**, eight positive person-level differences out
of nine, and exact two-sided sign-flip p = **0.01171875**. The absolute equal-person
mean R² is **−1.590746 for the frozen core and −3.108407 for scratch**. Every person's
future-day mean R² is negative under both methods. The contrast measures relative
protection against a poor scratch baseline; it does not establish useful decoding
accuracy or successful practical generalization. The fixed recipe was not retuned
after these results. Calibration curves use the same eight people across all three
budgets, while the primary 120-second paired contrast retains all nine.

The receipt binds these exact SHA256 identities:

| Artifact | SHA256 |
| --- | --- |
| Run manifest | `236127dcbe2c456b6826f09d52d7c4edbd9111f4c2f8a67fa138d17181a7555a` |
| Prepared index | `91de850304c4fb96096daa3f9ec19edab0372675c236a899005cf4308ec6c523` |
| Frozen plan | `7e91115abb8c9d54c1b2865befc2cdf36a1e2786b129ff63fc6f8362f6266415` |
| Run summary | `814681091b46b093e6f02b727e06da9a04a2d3a414e07eb3cbab29a4fc704971` |
| Auditor source | `385c093686687269425a94e1839603e3cf106873f36367b538c59f607e890d68` |
| Public aggregate | `7d5885860a18381368839dc80d198365ca4e8e66e1a2b157ea80324f4151e82f` |

Hash consistency and numerical agreement do not prove exclusive training-data use
by themselves; source review and perturbation tests support that separate claim.
This audit does not independently reproduce optimizer updates, replay decoder
predictions, or validate upstream acquisition filter causality, runtime deadlines,
or closed-loop clinical performance.

After a passing receipt, the publication gate reads JSON metadata only:

```sh
PYTHONPATH=src .venv/bin/python -m neural_experiments.v2.public transfer \
  --run results/neural-v2/transfer-v1 \
  --audit results/neural-v2/transfer-audit-v1.json \
  --output neural_experiments/results/v2-transfer.json
```

For runtime evidence, replace `transfer` with `runtime` and supply the completed
runtime run and independent timestamp-audit receipt. Both exports require exact
manifest and summary hashes and refuse an existing output. Public exports retain
model, trace and source digests, calibration counts, per-session metrics, and
aggregate timing evidence. They omit numeric weights, recording samples, timing
arrays, private artifact locations, and installed-library paths. Runtime audit
scope explicitly preserves that independent timestamp reconstruction does not
independently replay the decoder or establish prediction correctness.

## Post-hoc random frozen-core control

After the original nine-person results were inspected, the original frozen-versus-
scratch contrast was recognized to mix source-learned information with the effect
of freezing the temporal core. Commit `8a21df3` froze one exploratory control before
its execution: random frozen cores at the original scratch initialization, all nine
people, seeds 0/1/2, 120-second calibration, and the same 40-epoch adapter/head
training, prefix normalization, burn-in, masks, and evaluation sessions. No settings
were searched or changed after control outcomes. This is a post-hoc comparison in
the same already-observed cohort, not a new confirmatory test.

The first attempt completed in 18.937 seconds of run wall time and passed a separate
verification pass covering all 27 models and 162 full traces. Exact initial-core
and trained-core equality, model and prediction digests, calibration budgets and
effective masks, original learned model/core identities, prefix normalization,
and matched evaluation masks were checked. The run and verification receipts are
`results/neural-v2/random-core-v1` and `results/neural-v2/random-core-audit-v1.json`;
the public aggregate is `neural_experiments/results/v2-random-core.json`. No failed
attempt or retuning occurred. The source/plan and every run identity remain bound
in the receipts.

Future-day equal-person mean R² is **−0.562298 for random frozen cores**, compared
with **−1.590746 for learned frozen cores** and **−3.108407 for scratch**. Random
frozen cores outperform learned frozen cores in all nine person-level means. The
learned-minus-random mean difference is **−1.028448**, with a descriptive paired-
person bootstrap interval of **[−1.543836, −0.582877]**. Every person's future-day
mean remains negative under all three methods. These results weaken attributing
the original frozen-versus-scratch gain to learned source information; they do not
establish useful absolute accuracy. The common same-session suffix has eight
eligible people: random mean **0.023045**, learned mean **−0.079275**, and difference
**−0.102321**, with descriptive interval **[−0.240711, 0.022938]**. T6's short first
session remains explicitly unavailable for this suffix.

The control runner and its second verification pass share `random_core_control.py`
and its aggregation function. Both use the separately implemented existing
`audit.score_arrays` sklearn reference rather than the production scorer. This is
shared-source artifact/score verification with an independent numerical reference,
not an independent implementation of the control's entire analysis, optimizer
retraining, or decoder replay. The public key is `score_verification`, and its
receipt explicitly preserves those limits. The accompanying `v2-random-core`
PNG/SVG uses public aggregates only and shows all nine person means alongside all
three overall means.
