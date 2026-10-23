# Reproducing V6

Read the [results report](REPORT.md), [locked protocol](PROTOCOL.md), [data provenance](DATA.md), and
[exploratory T2 diagnosis](DIAGNOSIS.md) before interpreting results. This is an
offline replay on public BrainGate recordings, with known participants and an
unchanged direction-proxy target. The V6 controller confirms the exact exported model and can use strictly earlier learning curves to raise its training budget.

Use the repository's installed Python environment and prebuilt scientific
libraries. No Rust compilation is needed. Set `PYTHONPATH=.:src`,
`OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.

```sh
python -m pytest -q tests/neural_decoder/test_v6_*.py
python -m neural_experiments.v6.data --help
python -m neural_experiments.v6.study --help
python -m neural_experiments.v6.public --help
```

The public repository contains code and aggregates. Reproducing the real-data
study requires independently acquiring the pinned public provider release and
retaining the earlier V2–V5 parent manifests. Transport redirects remain local.
The V6 development index reuses exact prepared parent bytes; do not rewrite the
hard-linked sample files. Data preparation rejects a changed plan, changed
parents, overwritten outputs, or substituted session inventory.

The study entry points are:

```sh
python -m neural_experiments.v6.study select \
  --development-index data/neural_v6/development/index.json \
  --output results/neural_v6/source

python -m neural_experiments.v6.audit \
  --selection results/neural_v6/source/selection.json \
  --output results/neural_v6/source-audit.json
python -m neural_experiments.v6.public \
  --artifact results/neural_v6/source/selection.json \
  --audit results/neural_v6/source-audit.json \
  --output neural_experiments/results/v6-source-selection.json

# Publish the passed source choices before this assessment.
python -m neural_experiments.v6.study assess \
  --selection results/neural_v6/source/selection.json \
  --audit results/neural_v6/source-audit.json \
  --new-index data/neural_v6/new_sessions/index.json \
  --output results/neural_v6/assessment

python -m neural_experiments.v6.audit \
  --assessment results/neural_v6/assessment/assessment.json \
  --output results/neural_v6/assessment-audit.json
python -m neural_experiments.v6.public \
  --artifact results/neural_v6/assessment/assessment.json \
  --audit results/neural_v6/assessment-audit.json \
  --output neural_experiments/results/v6-study.json
```

Output directories are immutable attempts. Preserve failed attempts and create
new directories rather than silently overwriting evidence. The assessment
requires an exact passed source-audit receipt and unchanged execution-source
hashes. Once assessment outcomes have been seen, changes require a new study and
fresh holdout; rerunning on these outcomes does not restore freshness.

The predetermined streaming benchmark accepts the audited assessment manifest,
not an arbitrary model path. Run it after other local experiments have stopped:

```sh
python -m neural_experiments.v6.runtime \
  --assessment results/neural_v6/assessment/manifest.json \
  --index data/neural_v6/new_sessions/index.json \
  --plan neural_experiments/v6/plan.json \
  --assessment-audit results/neural_v6/assessment-audit.json \
  --output results/neural_v6/runtime
python -m neural_experiments.v6.runtime_audit \
  --run results/neural_v6/runtime \
  --output results/neural_v6/runtime-audit.json
python -m neural_experiments.v6.runtime_public \
  --run results/neural_v6/runtime \
  --audit results/neural_v6/runtime-audit.json

MPLCONFIGDIR=work/v6/matplotlib python -m neural_experiments.v6.figures \
  --study neural_experiments/results/v6-study.json \
  --output neural_experiments/figures
```

The exact original run is bound to its local, hash-pinned parent manifests.
Reacquiring or relocating data creates a new manifest lineage; matching the
public provider/member/array hashes is necessary, and silently changing the
original run's identity is not a valid reproduction. The published synthetic
suite runs without any private cache.
