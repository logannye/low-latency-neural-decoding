# H1 scoring audit

This project uses the H1 regression semantics from
[`snel-repo/falcon-challenge@770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f`](https://github.com/snel-repo/falcon-challenge/tree/770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f).
Local calculations and synthetic parity tests do not constitute an official FALCON
submission or score. Evaluation on public calibration recordings must be labeled
**local evaluation on public FALCON calibration recordings**. No sealed recordings
were opened or scored to build these fixtures.

## Information boundary and public API

`src/neural_decoder/metrics.py` is evaluator-only. `FileEvaluation` contains
`file_id`, audited `day`, full `predictions[T,D]`, full `targets[T,D]`, and a
boolean `eval_mask[T]`. The scorer never calls a predictor. Features, labels, and
masks must be separated by the replay/experiment layer before scoring; having this
module alone does not prove that an entire pipeline is free of leakage.

```python
from neural_decoder.metrics import FileEvaluation, score_files

metrics = score_files(
    [FileEvaluation(file_id, day, full_predictions, full_targets, evaluation_mask)]
)
```

The data loader fixes H1's seven-dimensional output order. The metric functions
allow any consistent positive output dimension so smaller exact-answer fixtures
remain easy to inspect. They do not infer anatomical output labels.

`score_files` returns a JSON-safe dictionary with:

- `r2_mean`, `r2_population_std`, `r2_worst_day`, and `worst_day`.
- `per_day[day]`: variance-weighted and per-output R², per-output squared-error
  sums and target sums of squares, zero-variance output indices, file identities,
  full/scored/unscored bin counts, and nominal durations.
- Global counts, `canonical_day_order`, `chronological_day_order`, the source pin,
  and `evaluation_kind: local_evaluation`.

Durations count bins at the declared nominal interval (default 0.02 seconds).
They are not an estimate of elapsed recording time spanning timestamp gaps.
Counts and durations do not establish the independence of observations.

## Direct calculation

For each day, concatenate that day's scored rows across all files before computing
the target mean. For output `j`, calculate

```
SSE[j] = sum((target[t,j] - prediction[t,j])**2)
SST[j] = sum((target[t,j] - mean_t(target[:,j]))**2)
R2[j]  = 1 - SSE[j] / SST[j], if SST[j] > 0
```

The day score is the `SST`-weighted average of output R². The primary score is
the arithmetic mean of day scores, giving every day equal weight. The reported
standard deviation is the population standard deviation (`ddof=0`), matching the
official helper. It is not a confidence interval or a standard error. File scores
are not averaged to obtain day scores, and days are not weighted by bin counts.
Negative R² values remain negative.

The implementation computes in float64, independently of scikit-learn. The
upstream helper calls `sklearn.metrics.r2_score(..., multioutput='variance_weighted')`
with default `force_finite=True`: an exactly constant output scores 1 when
predicted perfectly and 0 otherwise. Constant outputs receive zero weight if at
least one output varies. When every output is constant, the score falls back to
the uniform mean of those finite per-output scores. A large error on a constant
output can therefore be absent from the aggregate; per-output values and the
zero-variance indices are retained to disclose that behavior.

Our completed-report contract fails explicitly for fewer than two scored rows in
a day; upstream would return an undefined score for a one-row day. An all-masked
day also fails. A fully unscored file is retained when its day has enough scored
rows in other files. Shape mismatches, duplicate file IDs, non-boolean masks,
nonfinite scored targets, nonfinite predictions in any bin, and arithmetic
overflow fail. Unlike the upstream scorer, which only sees scored predictions,
the local wrapper also rejects a failed prediction on an unscored bin. Unscored
target values are ignored, including nonfinite placeholders.

## Canonical assembly and upstream interface traps

The pinned `evaluate` function iterates the submission dictionary in insertion
order. Its regression helper independently sorts day keys lexicographically.
Consequently, arrays submitted in chronological order can be partitioned at the
wrong offsets, particularly when `S6` and `S10` have different lengths. Scoring
uses lexical day order (`S10`, `S11`, `S12`, `S6`, ...), then lexical file ID order.
This canonical scoring order is separate from predictor replay order.

`assemble_official_regression(records)` returns an `OfficialRegressionInputs`
instance containing full predictions, **already-masked targets**, a full boolean
mask, full lengths grouped by day, and the canonical file order. The target mask
is applied exactly once. The upstream helper masks predictions internally and
adjusts cumulative full-array boundaries by the number of excluded rows. Supplying
full targets to that helper, pre-masking predictions, or constructing lengths
from scored rows violates its contract.

The documented `FalconEvaluator.evaluate` path constructs payloads and delegates
to the module-level `evaluate` function. The stale convenience methods
`evaluate_files` and `compute_metrics` have mismatched return/argument contracts
at this pin, so they are not integration entry points. The module-level wrapper
also expects a complete official held-in or held-out set; our local development
folds deliberately use the audited direct scorer on their declared subset.

## Offline reference and verification

The fixture `tests/neural_decoder/fixtures/falcon_evaluator_770c9b9.py.txt` is an
unchanged copy of upstream `falcon_challenge/evaluator.py`, SHA-256:

```
2b848f84ef620eac82ce53d331d03727d9ae4ea9a5f2e5edb6e6557537ce4418
```

`FALCON_LICENSE.txt` alongside it retains the upstream MIT license and copyright.
Tests verify the exact hash and extract the actual `evaluate`, `reduce_key`,
maps, and regression helper using Python's AST. Only those nodes execute, so the
offline test does not require installing the FALCON package or its unrelated
hydra, NWB, edit-distance, or torch imports. Pickle inputs to this reference are
created inside the test from synthetic arrays; external pickles are not loaded.

Known-answer tests check grouping before averaging, unequal day lengths, scattered
masks, double-mask rejection, zero-variance behavior, negative scores, counts,
and explicit invalid-input failures. Wrapper parity covers all 14 official
held-out file IDs with synthetic data, including unequal-length `S6`/`S10`. The
same fixture deliberately assembles chronological arrays against lexical lengths
and confirms that the resulting wrong score differs, demonstrating that the
ordering test can expose the known failure.

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/neural_decoder/test_h1_metrics.py -q
```

Initial focused verification: 18 tests passed on Python 3.12.13, NumPy 2.5.2,
scikit-learn 1.9.1, and pytest 9.1.1. The upstream setup does not pin scikit-learn;
that dependency identity is therefore part of parity-test evidence, not implied
by the FALCON source commit alone. Routine CI runs the tests with its recorded
installed versions. Numerical parity tests use float64 inputs and 1e-12 absolute
tolerance for wrapper aggregates; these are correctness fixtures, not biological
evaluation or deployment performance measurements.
