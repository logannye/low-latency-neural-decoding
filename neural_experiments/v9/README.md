# Reproduce trial-balanced calibration

V9 tests whether recruiting enough complete, directionally diverse trials and using a provider-informed training target can rescue T2. Twelve fixed arms distinguish extra label time, trial representation, direction weighting and target definition. The original continuous endpoint and identical evaluation mask determine recovery. Read the [protocol](PROTOCOL.md) and [data contract](DATA.md) first.

Use Python 3.12 with the repository's prebuilt scientific dependencies. No Rust build is required. Raw recordings, context, model coefficients and prediction traces remain in ignored local directories. Public evidence contains aggregate scores, costs, dispositions and hashes.

## Exposed source data and frozen code

The exact V8 development and assessment indexes are required. Their 200 previously exposed prefixes are reused without extensions; the four V9 participants supply 96 model-development sessions.

```sh
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m neural_experiments.v9.data prepare-development \
  --cache data/neural_v9/development
python -m neural_experiments.v9.audit_data \
  --index data/neural_v9/development/index.json \
  --output neural_experiments/results/v9-development-data-audit.json
```

The acquisition plan must be committed before downloads. The scientific plan and every producer dependency must be committed before source fits. Output directories are immutable: use a different directory for a reproduction and preserve original evidence. Reproducing already exposed outcomes does not create new held-out evidence.

## Source choice and independent seal

Four fixed reaction-offset/ridge combinations are compared on nine exposed T2 sessions. The primary method is fixed in advance; source selection chooses only these two hyperparameters. Every candidate and all twelve final methods are retained.

```sh
python -m neural_experiments.v9.development \
  --index data/neural_v9/development/index.json \
  --output results/neural_v9/development \
  --plan-sha256 aea60063bf6aa027ea54ce05dfb52e83ff00c61f8728f3a8f59f4f5394431dbf
python -m neural_experiments.v9.audit \
  --run results/neural_v9/development \
  --data-audit neural_experiments/results/v9-development-data-audit.json \
  --output neural_experiments/results/v9-source-audit.json
python -m neural_experiments.v9.public \
  --manifest results/neural_v9/development/manifest.json \
  --audit neural_experiments/results/v9-source-audit.json \
  --output neural_experiments/results/v9-development-study.json \
  --selection-output neural_experiments/v9/source-selection.json
```

Commit the passed source audit, aggregate source results and sealed selection before fresh conversion. The guard checks committed plan, source bytes, audit identity and exact selected recipe. A raw download alone does not authorize opening fresh arrays.

## Fixed fresh cohort

Archive ranks 25–30 are fixed for T2, T7, T9 and T10. All 24 earlier members must match the known parents. The importer retains failed, missing or short cases without substitutions. Transfer and storage limits include failed attempts. Provider-issued download redirects may be supplied using `--url-file` with an ignored private file; this does not bypass provider verification.

```sh
python -m neural_experiments.v9.acquire_parallel \
  --cache data/neural_v9/fresh --workers 3
python -m neural_experiments.v9.prepare_fresh \
  --cache data/neural_v9/fresh \
  --selection neural_experiments/v9/source-selection.json \
  --selection-sha256 "$(shasum -a 256 neural_experiments/v9/source-selection.json | cut -d ' ' -f 1)"
python -m neural_experiments.v9.audit_data \
  --index data/neural_v9/fresh/index.json \
  --output neural_experiments/results/v9-fresh-data-audit.json
python -m neural_experiments.v9.cohort_public
```

Only the first 600 recorded seconds are prepared. Each fresh session uses its own purchased labels. No fresh session trains another session or alters the source choice.

## One-shot assessment and publication

```sh
python -m neural_experiments.v9.assessment \
  --index data/neural_v9/fresh/index.json \
  --output results/neural_v9/assessment \
  --selection neural_experiments/v9/source-selection.json \
  --selection-sha256 "$(shasum -a 256 neural_experiments/v9/source-selection.json | cut -d ' ' -f 1)"
python -m neural_experiments.v9.audit \
  --run results/neural_v9/assessment \
  --data-audit neural_experiments/results/v9-fresh-data-audit.json \
  --source-run results/neural_v9/development \
  --source-audit neural_experiments/results/v9-source-audit.json \
  --output neural_experiments/results/v9-assessment-audit.json
python -m neural_experiments.v9.public \
  --manifest results/neural_v9/assessment/manifest.json \
  --audit neural_experiments/results/v9-assessment-audit.json \
  --output neural_experiments/results/v9-study.json
python -m neural_experiments.v9.plot \
  --study neural_experiments/results/v9-study.json \
  --output neural_experiments/figures/v9-trial-calibration.png
```

The numerical auditor reconstructs quota stopping, trial windows, targets, weights, fits, saved model predictions, primary/secondary scores, source selection, paired uncertainty and both recovery gates. It does not import the V9 model or metrics producer. The raw-data audit separately validates the context and unchanged primary targets. Hash identity supports reproducibility; it does not establish biological intention or clinical effectiveness.

## Software checks

```sh
python -m pytest -q tests/neural_decoder
ruff check src/neural_decoder tests/neural_decoder neural_experiments
ruff format --check src/neural_decoder tests/neural_decoder neural_experiments
```

Synthetic checks exercise held-out-label perturbations, incomplete trials, target changes, quota failures, block resets, causal streaming, export parity, numerical reconstruction, publication privacy and corrupted-artifact rejection. Hosted CI runs the complete Python suite on Linux. Existing V8 latency measurements refer to that exported decoder and are not new V9 timing results.
