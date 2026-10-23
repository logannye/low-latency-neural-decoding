# Reproduce the T2 mechanism study

V8 separates three proposed failure mechanisms: the target definition, calibration trial coverage, and neural feature representation. It retains the original continuous direction endpoint, evaluates a fixed factorial design on fresh sessions, and keeps target changes as separate diagnostic endpoints. Read the [protocol](PROTOCOL.md), [data contract](DATA.md), and [results](REPORT.md) before comparing these numbers with previous versions.

Use the repository's Python environment with prebuilt NumPy/SciPy libraries. No Rust compilation is required. Raw recordings, behavioral context, fitted coefficients, prediction traces, and transport links remain in ignored `data/`, `results/`, and `work/` directories. Public JSON files contain allowlisted aggregate results and provenance.

## 1. Prepare the exposed development cohort

Existing V7 parent indexes and their authenticated raw recordings are required. The importer preserves their exposed prefixes and reuses the original neural features, targets, and masks.

```sh
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m neural_experiments.v8.data prepare-development \
  --cache data/neural_v8/development
python -m neural_experiments.v8.audit_data \
  --index data/neural_v8/development/index.json \
  --output neural_experiments/results/v8-development-data-audit.json
```

The [data guide](DATA.md) describes preparation and its exact parent hashes. Loading compressed MATLAB containers may materialize their variables internally; numerical operations and exports remain bounded to the authorized prefix. The raw-data audit reconstructs the context and original targets from those selected source segments.

## 2. Reproduce source selection

The source runner requires the study plan to be committed in this repository and uses immutable output directories. Choose a new output directory for a rerun; preserve the original evidence.

```sh
python -m neural_experiments.v8.development \
  --index data/neural_v8/development/index.json \
  --output results/neural_v8/development \
  --plan-sha256 01fc7756fe3110630a98f78152a08f85b2119d76c211196da93beb23f405a392
python -m neural_experiments.v8.audit \
  --run results/neural_v8/development \
  --data-audit neural_experiments/results/v8-development-data-audit.json \
  --output results/neural_v8/development-audit.json
python -m neural_experiments.v8.public \
  --manifest results/neural_v8/development/manifest.json \
  --audit results/neural_v8/development-audit.json \
  --output neural_experiments/results/v8-development-study.json \
  --selection-output neural_experiments/v8/source-selection.json
```

T2 archive ranks 1–15 choose a representation from eight fixed options. Ranks 16–21 choose one of eight combinations of representation, training phase, and trial weighting. All 21 sessions were already exposed in earlier work; this is development, not a new validation claim. The final source choice, [source-selection.json](source-selection.json), is sealed only after the independent numerical audit passes. Every candidate result is retained.

Review and commit the newly audited selection before fresh preparation or assessment:

```sh
git add neural_experiments/v8/source-selection.json neural_experiments/results/v8-development-study.json
git commit -m 'Seal independently audited V8 source choices'
git push
```

The assessment checks the selection against committed bytes, so an uncommitted rerun will be rejected.

## 3. Acquire and prepare the fixed fresh cohort

Acquisition streams the public archives, authenticates all 21 previous members for each participant, and retains ranks 22–24. The committed acquisition plan fixes the participants, ranks, transfer budget and storage budget before downloads. It never substitutes a longer or better recording. Rank order is not chronological day order.

```sh
python -m neural_experiments.v8.acquire_parallel \
  --cache data/neural_v8/fresh --workers 3
python -m neural_experiments.v8.fresh_data prepare \
  --cache data/neural_v8/fresh \
  --study-plan neural_experiments/v8/study-plan.json \
  --study-plan-sha256 01fc7756fe3110630a98f78152a08f85b2119d76c211196da93beb23f405a392
```

If the provider's normal public flow has issued download redirects, `--url-file` accepts the private mapping. It is never committed. Complete any provider verification through its normal browser flow; the code does not bypass verification. Fresh conversion requires a committed study plan and source choices frozen before outcome access.

## 4. Run the one-shot assessment and independent audits

```sh
python -m neural_experiments.v8.assessment \
  --index data/neural_v8/fresh/index.json \
  --output results/neural_v8/assessment \
  --selection neural_experiments/v8/source-selection.json \
  --selection-sha256 "$(shasum -a 256 neural_experiments/v8/source-selection.json | cut -d ' ' -f 1)"
python -m neural_experiments.v8.audit_data \
  --index data/neural_v8/fresh/index.json \
  --output neural_experiments/results/v8-fresh-data-audit.json
python -m neural_experiments.v8.audit \
  --run results/neural_v8/assessment \
  --data-audit neural_experiments/results/v8-fresh-data-audit.json \
  --source-run results/neural_v8/development \
  --source-audit results/neural_v8/development-audit.json \
  --output results/neural_v8/assessment-audit.json
```

The assessment refuses altered scientific code or unsealed choices. It evaluates all eight factorial cells and four controls, preserving every case. All primary methods share the same scoring mask, including exclusion of every trial that overlaps purchased calibration. The auditor separately reconstructs fits, predictions, scoring, source choices, confidence intervals and the recovery gate.

Publication uses `v8.public`, which requires a matching passed audit. `v8.cohort_public` publishes acquisition and cohort provenance. `v8.diagnostics` reports target geometry, trial coverage and neural sparsity. `v8.errors` decomposes the saved prediction errors. `v8.ceiling` is explicitly a post-development oracle diagnostic: its per-session scale uses held-out targets and cannot be presented as deployable accuracy.

The error decomposition and scalar-oracle helpers omit their diagnostic R² when per-row target variance is at most `1e-12`. The authoritative scorer instead guards total summed variance. These differ on nearly constant synthetic targets; none of the 576 source or 828 fresh decompositions is affected. The minimum fresh variance is 0.7777. Preserve authoritative scores and report any such unavailable diagnostic explicitly, rather than treating it as zero accuracy.

## 5. Check engineering behavior

```sh
python -m pytest -q tests/neural_decoder/test_v8*.py
python -m neural_experiments.v8.runtime --help
```

The [runtime protocol and result](RUNTIME.md) use the first factorial development session by archive rank, without choosing a favorable session or model after timing. Source replay checks exported batch/streaming parity and observed processing latency; it does not establish fresh decoding accuracy or a hard real-time bound. Scientific input filtering adds response lag separately from CPU service time.
