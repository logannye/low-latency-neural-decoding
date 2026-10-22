# Reproducing V5

Read the [locked protocol](PROTOCOL.md), [data provenance](DATA.md), and
[exploratory T2 diagnosis](DIAGNOSIS.md) before interpreting results. This is an
offline replay on public BrainGate recordings, with known participants and an
unchanged direction-proxy target.

Use the repository's installed Python environment and prebuilt scientific
libraries. No Rust compilation is needed. Set `PYTHONPATH=.:src`,
`OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.

```sh
python -m pytest -q tests/neural_decoder/test_v5_*.py
python -m neural_experiments.v5.data --help
python -m neural_experiments.v5.study --help
python -m neural_experiments.v5.public --help
```

The public repository contains code and aggregates. Reproducing the real-data
study requires independently acquiring the pinned public provider release and
retaining the earlier V2–V4 parent manifests. Transport redirects remain local.
The V5 development index reuses exact prepared parent bytes; do not rewrite the
hard-linked sample files. Data preparation rejects a changed plan, changed
parents, overwritten outputs, or substituted session inventory.

The study entry points are:

```sh
python -m neural_experiments.v5.study select \
  --development-index data/neural_v5/development/index.json \
  --output results/neural_v5/source

# Independently audit and publish source choices before this assessment.
python -m neural_experiments.v5.study assess \
  --selection results/neural_v5/source/selection.json \
  --audit results/neural_v5/source-audit.json \
  --new-index data/neural_v5/new_sessions/index.json \
  --t3-index data/neural_v5/t3_blocks/index.json \
  --output results/neural_v5/assessment

python -m neural_experiments.v5.public \
  --artifact results/neural_v5/assessment/assessment.json \
  --audit results/neural_v5/assessment-audit.json \
  --output neural_experiments/results/v5-study.json
```

Output directories are immutable attempts. Preserve failed attempts and create
new directories rather than silently overwriting evidence. The assessment
requires an exact passed source-audit receipt and unchanged execution-source
hashes. Once assessment outcomes have been seen, changes require a new study and
fresh holdout; rerunning on these outcomes does not restore freshness.
