# Reproduce V7 local recoverability

V7 first tests whether recent labels restore useful decoding on the already
exposed recordings. The [protocol](PROTOCOL.md) and
[locked plan](recovery-plan.json) define a development gate before any new
continual-policy or fresh-data work. A development pass is not a fresh result.

Use the existing Python scientific environment; no Rust compilation is needed.
Raw/prepared neural samples, model coefficients and full predictions remain in
ignored `data/`, `results/` and `work/` directories. Public evidence contains
aggregate metrics and identities only.

## Prepare the exposed union

The two V6 indexes must already match their published hashes. This reuses their
first-at-most-600-second feature and target files through immutable hard links;
it never opens raw MAT recordings or previously unexposed continuation tails.
Consult `python -m neural_experiments.v7.data --help` for preparation arguments.
The original prepared union has 152 sessions and 4,264,975 bins, with index SHA256
`4b2c0c8b47a2caf1a2135b28dcea197a26c8f6006c19f243ff2fed23c5e1362e`.
Repreparing changes index timestamps and local paths, not parent sample bytes.

## Execute and independently audit

Every output directory must be new. The recovery runner snapshots source files,
records their hashes and checks they did not change during execution. It exports
each model and checks prediction parity after loading it from disk.

```sh
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m neural_experiments.v7.recoverability \
  --index data/neural_v7/development/index.json \
  --plan neural_experiments/v7/recovery-plan.json \
  --plan-sha256 28ead298081c41b6028bffa0cf9aa5ee0eeb7811917400863aff350525aa7671 \
  --output results/neural_v7/recoverability
.venv/bin/python -m neural_experiments.v7.audit \
  --run results/neural_v7/recoverability \
  --output results/neural_v7/recovery-audit.json
.venv/bin/python -m neural_experiments.v7.public \
  --manifest results/neural_v7/recoverability/manifest.json \
  --output results/neural_v7/recovery-public.json
```

The auditor reconstructs the causal representation, fits, normalization,
predictions, scoring masks, costs, bootstrap summaries and gate without importing
the producer model, scorer or runner. This is an independent implementation
within the project, not external replication.

## Interpretation

Dense labels provide a diagnostic of the tested decoder family's recoverability.
They are not a deployable error monitor, and their cost is retained.
Label cost here is the union of intervals used by executed diagnostic fits;
sessions ending before the first forecast incur no fit or label cost. This is
retrospective data access accounting, not simulated prospective interruption time.
Expanding and rolling methods may reuse earlier scored labels only after the two-second
delay. Scoring windows are temporally disjoint; missing trial IDs prevent a
claim of independent trials. Whole-session and whole-person uncertainty units
avoid treating individual bins or repeated windows as independent replications.

If no declared recent-label method passes every gate, stop before fresh-data
acquisition and publish that result. If one passes, a separate locked study must
test sparse periodic and triggered calibration with matched label budgets,
including detection failures and calibration interruptions.
