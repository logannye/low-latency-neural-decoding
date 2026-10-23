# Reproduce V7 recoverability and continual recalibration

V7A tests whether recent labels restore useful decoding on exposed recordings.
Its [original protocol](PROTOCOL.md) and [locked plan](recovery-plan.json) remain
unchanged. The gate failed; read the [recovery report](RECOVERY_REPORT.md).
V7B follows an explicitly disclosed [change to the continuation rule](CONTINUAL_PROTOCOL.md)
to test accumulating updates and sparse feedback on fresh sessions. It does not
retroactively turn the V7A failure into a pass.

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

The original stopping rule prescribed no fresh-data work after a failed gate.
The separately committed V7B amendment explains why the accumulating-history
control motivated a new study despite that failure. Preserve both decisions
when reproducing or extending these experiments.

## V7B source selection and fresh assessment

The fixed V7B [plan](continual-plan.json) compares two-minute startup, four-minute
startup, periodic updates and neural-shift-triggered updates over one common
deployment interval. Calibration interruptions yield zero output and remain in
the score. Source threshold ranking excludes the entire target participant.

```sh
.venv/bin/python -m neural_experiments.v7.study_continual source \
  --index data/neural_v7/development/index.json \
  --output results/neural_v7/continual-source
.venv/bin/python -m neural_experiments.v7.audit_continual \
  --run results/neural_v7/continual-source \
  --output results/neural_v7/continual-source-audit.json
.venv/bin/python -m neural_experiments.v7.public_continual \
  --manifest results/neural_v7/continual-source/manifest.json \
  --audit results/neural_v7/continual-source-audit.json \
  --output neural_experiments/results/v7-continual-source.json \
  --selection-output neural_experiments/results/v7-continual-selection.json
```

Publish the passed source choices before scoring fresh sessions. Fresh data are
exactly regular archive ranks 19–21 from eight known participants in pinned Dryad
version 462702. The [acquisition plan](acquisition-plan.json) requires authenticating
the previous 18 members, retaining every selected case, and respecting total
transfer, local-storage and worker limits. A provider redirect may be supplied
locally when the provider API requires browser verification. Do not publish
transport grants or substitute sessions. Inspect the downloader and preparation
CLIs for the exact parent-index and transport arguments:

```sh
.venv/bin/python -m neural_experiments.v7.acquire_parallel --help
.venv/bin/python -m neural_experiments.v7.fresh_data --help
.venv/bin/python -m neural_experiments.v7.fresh_data prepare \
  --cache data/neural_v7/fresh \
  --study-plan neural_experiments/v7/continual-plan.json \
  --study-plan-sha256 4f88a6cf98dd7147691f4eaf64143784736863e1cfa4fd5fbd7b3fadd1eaea23
```

The assessment requires the exact SHA256 of the published selection file and
rejects changed scientific source. Use the selection digest recorded in the
final report for the original run; independently rebuilt data and choices have
their own lineage and digests.

```sh
.venv/bin/python -m neural_experiments.v7.study_continual assessment \
  --index data/neural_v7/fresh/index.json \
  --output results/neural_v7/continual-assessment \
  --selection neural_experiments/results/v7-continual-selection.json \
  --selection-sha256 "$V7_SELECTION_SHA256"
.venv/bin/python -m neural_experiments.v7.audit_continual --help
.venv/bin/python -m neural_experiments.v7.audit_continual \
  --run results/neural_v7/continual-assessment \
  --source-run results/neural_v7/continual-source \
  --source-audit results/neural_v7/continual-source-audit.json \
  --output results/neural_v7/continual-assessment-audit.json
.venv/bin/python -m neural_experiments.v7.public_continual \
  --manifest results/neural_v7/continual-assessment/manifest.json \
  --audit results/neural_v7/continual-assessment-audit.json \
  --output neural_experiments/results/v7-continual-study.json
```

The final auditor binds the fresh assessment to its passed source audit and
selection, reconstructs all fits, decisions, full traces and aggregate gates,
and authenticates the fixed cohort. A numerical audit pass does not imply the
scientific success gates passed. Only publish aggregates after that audit.

Recovery delay and pre-request error diagnostics have their own
[fixed definitions](DIAGNOSTICS.md). They use scorer targets retrospectively;
those targets never enter the controller. Run `diagnostics_continual --help`
and `runtime_continual --help` as Python modules for the descriptive and separate
source-model runtime measurements. Timing a fixed exported model over a prefix
is an engineering check, not an accuracy evaluation of that model on its past.

The separate [streaming calibration implementation](incremental.py) accumulates
centered sufficient statistics with fixed normalization. Its engineering replay
checks all five periodic fits against the original source artifacts. This path
does not change the frozen fresh-study implementation or tune its predictions.

```sh
.venv/bin/python -m neural_experiments.v7.incremental_runtime \
  --source results/neural_v7/continual-source/manifest.json \
  --index data/neural_v7/development/index.json \
  --source-audit results/neural_v7/continual-source-audit.json \
  --output results/neural_v7/continual-incremental
```

Read [engineering evidence](ENGINEERING.md) for timing scope, memory accounting,
independent audits and the difference between recorded compute observations and
a real-time guarantee.

Every run directory is an immutable attempt. Do not overwrite failed runs or
relabel exposed outcomes as fresh. Public synthetic tests need no data cache.
