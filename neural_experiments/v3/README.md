# Reproduce the V3 diagnosis and held-out session study

The [locked plan](plan.json), [protocol](PROTOCOL.md), and
[failure diagnosis](DIAGNOSIS.md) define the information boundaries. New sessions
come from the same previously studied people. Each daily-calibrated model uses
its own session's first 120 or 300 observed seconds; subsequent scoring labels
never enter fitting or source selection. This is a new-session evaluation of a
calibrated decoding procedure, not zero-shot transfer or a new-participant test.

Use Python 3.12 with prebuilt dependencies, from the repository root:

```sh
python -m pip install --only-binary=:all: -e '.[dev,neural,runtime,benchmark]'
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m pytest -q tests/neural_decoder/test_v3_*.py
```

The synthetic tests require no recordings. The published study reused installed
native libraries and performed no local Rust compilation. Choose new run/output
paths for a reproduction; preserve previous attempts and their evidence.

## Data

First reproduce the original 54-session V2 acquisition using its
[data guide](../v2/DATA.md). The V3 development index is an independent export of
only those sessions' original first 600 seconds, adding spike-band power to
threshold counts. Preparation never fits statistics.

```sh
python -m neural_experiments.v3.data prepare-development \
  --cache data/neural_v3/development --v2-index data/neural_v2/index.json
python -m neural_experiments.v3.data metadata --cache data/neural_v3/assessment
python -m neural_experiments.v3.data acquire \
  --cache data/neural_v3/assessment --v2-index data/neural_v2/index.json
python -m neural_experiments.v3.data prepare-assessment \
  --cache data/neural_v3/assessment --v2-index data/neural_v2/index.json
python -m neural_experiments.v3.data validate \
  --cache data/neural_v3/development --expected-role development
python -m neural_experiments.v3.data validate \
  --cache data/neural_v3/assessment --expected-role assessment
```

Acquisition retains regular MAT members 7–9 per person, authenticates the old
first six, and records any archive shortage. It does not replace short or poor
recordings. See [DATA.md](DATA.md) for limits, ordinary provider verification,
paired count/power semantics, and the exact feature/target separation.

## Select using only exposed development data

```sh
python -m neural_experiments.v3.study select \
  --development-index data/neural_v3/development/index.json \
  --output results/neural-v3/reproduction
python -m neural_experiments.v3.audit \
  --selection results/neural-v3/reproduction/selection.json \
  --development-index data/neural_v3/development/index.json \
  --output results/neural-v3/reproduction-source-audit.json
```

The finite grid has 36 combinations of feature family, causal time constant,
baseline correction, and regularization, evaluated at each calibration budget.
For an outer participant, all of that person's development scores are excluded
from ranking. Equal-session scores are averaged within each source person, then
people receive equal weight. Exact ties use grid order. Every feasible candidate
and unavailable case remains in the selection ledger. Save and publish the
selected settings before proceeding to assessment.

The estimator minimizes
`sum_i ||y_i - z_i W - b||² / n + lambda * ||W||²`, with a free intercept.
The denominator is the number of training bins, not twice that number. This
makes the regularization convention explicit despite two output coordinates
and different calibration durations. The original V2 ridge comparator retains
its sum-loss `alpha=10` convention and finite filter; its calculation uses
float64 in V3 and is recipe-matched, not a bitwise reproduction of V2 float32.

## Assess once, then independently verify

```sh
python -m neural_experiments.v3.study assess \
  --run results/neural-v3/reproduction \
  --assessment-index data/neural_v3/assessment/index.json \
  --selection-audit results/neural-v3/reproduction-source-audit.json
python -m neural_experiments.v3.audit \
  --run results/neural-v3/reproduction \
  --development-index data/neural_v3/development/index.json \
  --assessment-index data/neural_v3/assessment/index.json \
  --output results/neural-v3/reproduction-assessment-audit.json
```

The runner refuses assessment without a passed, hash-bound source-selection
audit. It also rejects changes to its plan or executed model/study/data sources
after selection. The assessment transition is one-shot: it preserves failures
and refuses to restart a partially exposed run under the same destination.
The normal path reads new labels only after source selection is fixed; each fit
receives only its permitted prefix. Models are exported and reloaded before
feature-only prediction. Every mask and time range is retained.

The independent auditor reconstructs all source and assessment scores, checks
every planned case and missing disposition, reconstructs prefix normalization,
and verifies the ridge normal equations and free intercept. It independently
replays assessment predictions using separate filter recurrences and exported
coefficients. Source-grid fits are not independently retrained. Source hashes
alone do not prove information isolation; code review and boundary-mutation
tests provide additional evidence.

## Publish aggregate evidence and figures

```sh
python -m neural_experiments.v3.public selection \
  --input results/neural-v3/reproduction/selection.json \
  --audit results/neural-v3/reproduction-source-audit.json \
  --output work/reproduced-source-selection.json
python -m neural_experiments.v3.public study \
  --input results/neural-v3/reproduction \
  --audit results/neural-v3/reproduction-assessment-audit.json \
  --output work/reproduced-v3-results.json
python -m neural_experiments.v3.figures \
  --results work/reproduced-v3-results.json --output work/reproduced-v3-figures
```

Only aggregate scores, source identities, hashes, model settings, case counts,
and metadata are published. Raw neural arrays, coefficient exports, training
labels, sample predictions, signed transport URLs, and timing timestamps remain
in ignored caches. The public exporter requires a passed audit bound to the
exact run. Output paths under `work/` above avoid overwriting published evidence.

## Exact-artifact runtime

The runtime representative is selected without its score: overall-selected
120-second model, first eligible person in plan order, then first selected
regular archive-member rank. Use that identity's model and its session feature
file, never the best-looking or fastest export.

```sh
python -m neural_experiments.v3.runtime \
  --model PATH_FROM_MODEL_IDENTITY \
  --features PATH_FROM_ASSESSMENT_INDEX \
  --plan neural_experiments/v3/plan.json \
  --output results/neural-v3/reproduction-runtime
python -m neural_experiments.v3.audit_runtime \
  --run results/neural-v3/reproduction-runtime \
  --output results/neural-v3/reproduction-runtime-audit.json
python -m neural_experiments.v3.public runtime \
  --input results/neural-v3/reproduction-runtime \
  --audit results/neural-v3/reproduction-runtime-audit.json \
  --output work/reproduced-v3-runtime.json
```

Run timing after fitting, downloads, and other substantial background work have
stopped. Full-workload batch/stream parity precedes three fast-replay repeats and
two paced 60-second blocks. Processing includes validation, transforms, filter
state, normalization, the linear head, output projection, and block reset.
Report scheduling and release-to-completion separately. A causal exponential
filter has no future lookahead, but smoothing still adds response lag: 100,
200, and 400 ms time constants have approximately 90, 190, and 390 ms steady-state
mean sample age. This temporal response is distinct from CPU processing time.
Upstream feature causality, acquisition latency, and hard real-time guarantees
remain unverified.
