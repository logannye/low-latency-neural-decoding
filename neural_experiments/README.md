# Reproducing the neural decoding study

The [reviewer guide](REVIEWER_GUIDE.md) connects the positive H1 and V3/V4 results
to their exact experiments and retained failures. The [V9 study](v9/README.md) is
the latest calibration experiment; its trial-balanced policy did not recover T2.
The [final H1 deployment protocol](H1_FINAL_CLOSURE_PROTOCOL.md) specifies the
separate conversion and runtime validation of the existing final H1 models.

The [V2 study](v2/README.md) adds controlled H1 accuracy experiments and a
separate nine-participant calibration and transfer evaluation. This page
preserves the original V1 reproduction procedure and its evidence boundary.

The H1 study is complete: forward-day development, the two bounded research
hypotheses, frozen S5 and S6–S12 assessments, same-weight runtime measurements,
separate supervised-prefix calibration, independent audits, and a private replay
viewer. The selected 23,367-parameter TCN averaged **0.177583 day R²** over three
seeds on S6–S12, versus **0.034146** for ridge. The compression experiment and
small M2 recipe-transfer check are separate from that primary result. Read the
[final report](FINAL_REPORT.md) and [aggregate evidence](results/) for outcomes,
negative results, and limits.

These are local evaluations on public recordings from one H1 participant, not
official FALCON scores, cross-participant validation, or live neural control.
Runtime uses separately pinned development weights on one M4 Max; timing does not
include neural acquisition or wireless transport and does not establish implant
power savings. New machines, library versions, training artifacts, and freezes
produce new evidence rather than inheriting the original run's acceptance.

## Environment and synthetic checks

Run the commands below from the repository root in one Bash session. Use Python
3.12 and reuse a compatible existing `.venv`; create it only if none exists.
Dependencies come from prebuilt wheels. The editable project has no native
extension or Rust build. The `benchmark` extra supplies plotting dependencies;
`runtime` supplies the pinned ONNX and ONNX Runtime versions.

```sh
set -euo pipefail
if [ ! -x .venv/bin/python ]; then
  python3.12 -m venv .venv
fi
source .venv/bin/activate
python --version
python -m pip install --only-binary=:all: -e '.[dev,neural,runtime,benchmark]'
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XDG_CACHE_HOME="$PWD/data/neural_h1/library_cache"
export MPLCONFIGDIR="$PWD/work/matplotlib"

env -u NEURAL_H1_INDEX python -m pytest -q tests/neural_decoder
python -m neural_decoder.cli --help

export H1_CACHE="data/neural_h1"
export H1_INDEX="$H1_CACHE/prepared/index.json"
export H1_RUN="results/neural-h1/reproduction-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p results/neural-h1
mkdir "$H1_RUN"
python neural_experiments/replay_demo.py --smoke \
  --runtime-summary neural_experiments/results/runtime.json \
  --output "$H1_RUN/synthetic-replay.html"
```

The synthetic tests and smoke HTML need no neural download. Open the HTML locally;
its seven curves are deterministic synthetic data, not a decoding result. Every
experiment below uses a new child directory. Preserve failed attempts and choose
a fresh run name for retries; do not overwrite models, predictions, or audit
receipts. Keep the shell variables above for the remaining commands.

## Acquire and prepare the pinned H1 data

Read the [provider terms, attribution, and schema](data/README.md) before the
first download. Downloading constitutes agreement to H1's additional data-use
terms, including its restriction on redistribution. Keep raw recordings, derived
sample arrays, trained artifacts, and recording-derived replay files local in the
ignored cache/results directories. Use one raw/source cache, not a second copy
for each experiment. The 40 pinned raw assets total 102,258,934 bytes.

For a new cache:

```sh
python -m neural_decoder.data acquire --cache "$H1_CACHE" \
  --manifest neural_experiments/data/h1_manifest.json
python -m neural_decoder.data prepare --cache "$H1_CACHE"
NEURAL_H1_INDEX="$H1_INDEX" python -m pytest -q \
  tests/neural_decoder/test_h1_data.py tests/neural_decoder/test_splits.py
```

Acquisition verifies exact asset UUIDs, sizes and provider/local SHA-256 values,
and caches the required upstream files at commit
`770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f`. It does not install the full challenge
package. A changed draft cannot substitute different asset bytes. If this exact
cache and prepared index already exist, reuse them and run the checks; do not
reprepare during an experiment because preparation can rewrite the index/arrays.

Preparation inspects schema metadata for every file and exports only the 11
S0–S4 calibration streams: 112,488 bins, 176 channels, seven outputs. It preserves
the pinned official loader's timing/binning and checks exact array parity.
Feature NPZs contain only counts, timestamps, sequence IDs and channel IDs;
targets, masks and trial changes are separate scorer files. All predictors reset
per file and continue through trials/unscored bins. Minival overlaps calibration
and is excluded from H1 model selection and final assessment.

## Development and the two fixed hypotheses

The development folds are S0–S1→S2, S0–S2→S3 and S0–S3→S4. First reproduce the
six-setting ridge grid and original three-seed TCN. The optional pilot executes
one fold/seed for cost checking; it is a separate output, not a final result.

```sh
python -m neural_decoder.cli baseline --index "$H1_INDEX" \
  --config neural_experiments/configs/ridge-v1.json \
  --output "$H1_RUN/ridge"
python neural_experiments/audit.py --run "$H1_RUN/ridge" \
  --index "$H1_INDEX" --output "$H1_RUN/ridge/independent-audit.json"

python -m neural_decoder.cli temporal --index "$H1_INDEX" \
  --config neural_experiments/configs/tcn-v1.json \
  --output "$H1_RUN/tcn-pilot" --pilot
python -m neural_decoder.cli temporal --index "$H1_INDEX" \
  --config neural_experiments/configs/tcn-v1.json \
  --output "$H1_RUN/tcn"
python neural_experiments/audit_temporal.py --run "$H1_RUN/tcn" \
  --index "$H1_INDEX" --output "$H1_RUN/tcn/independent-audit.json"

python -m neural_decoder.run_adaptation --index "$H1_INDEX" \
  --baseline-run "$H1_RUN/ridge" \
  --config neural_experiments/configs/adaptation-v1.json \
  --output "$H1_RUN/adaptation"
python neural_experiments/audit_adaptation.py --run "$H1_RUN/adaptation" \
  --index "$H1_INDEX" --baseline-run "$H1_RUN/ridge" \
  --output "$H1_RUN/adaptation/independent-audit.json"
python -m neural_decoder.train_augmented --index "$H1_INDEX" \
  --baseline-run "$H1_RUN/tcn" \
  --config neural_experiments/configs/tcn-dropout-v1.json \
  --output "$H1_RUN/tcn-dropout"
python neural_experiments/audit_temporal.py --run "$H1_RUN/tcn-dropout" \
  --index "$H1_INDEX" --output "$H1_RUN/tcn-dropout/independent-audit.json"
```

[ADAPTATION.md](ADAPTATION.md) specifies the causal predict-before-update mean
correction and retrospective stress conditions. Its locked protective-bound rule
failed, so the completed study retained frozen normalization despite ordinary
EMA's development benefit. [AUGMENTATION.md](AUGMENTATION.md) specifies the single
p=0.1 training channel-dropout comparison. It passed the seed-averaged keep rule;
individual seed/day harms remain reported. There was no best-seed selection or
subsequent parameter search.

For this reproduction, retain [the selected frozen recipe](configs/frozen-recipe-v1.json)
and all three seeds. Recomputed development decisions remain evidence to report;
do not silently redesign the final recipe using the published assessment scores.
The independent ridge/TCN audits reconstruct saved predictions and scores. The
adaptation auditor uses separate delayed-EMA and batch-affine arithmetic to
check retained prediction traces, final EMA states and sklearn scores across all
conditions. The runner also retains its decision and artifact hashes; synthetic
contract tests supplement these checks.

A standalone feature-only replay does not load scorer labels:

```sh
python -m neural_decoder.cli replay --kind tcn \
  --model "$H1_RUN/tcn/seed-0/S2/model.npz" \
  --features "$H1_CACHE/prepared/features/S2_set_1.npz" \
  --output "$H1_RUN/feature-only-replay-S2.npy"
```

## Same-weight runtime reproduction

Read [RUNTIME_PROTOCOL.md](RUNTIME_PROTOCOL.md) for the service boundary,
500-bin warmup/reset, fixed input order, three fast repeats and two 60-second
paced blocks per implementation. Stop training, audits, browsing and other heavy
work during timing. Use one measurement process. Profiling is a separate
instrumented diagnostic; it is not a latency measurement.

The runtime workload uses the **original** TCN's preselected seed 0/S4 model,
not the later dropout refit. The published runtime config pins the original
artifact's SHA-256. A newly trained artifact needs its own config identity even
when its numeric predictions match. Create a new config, changing only that model
hash; do not edit the published config to impersonate its historical run.

```sh
export H1_RUNTIME_MODEL="$H1_RUN/tcn/seed-0/S4/model.npz"
export H1_RUNTIME_CONFIG="neural_experiments/configs/runtime-reproduction-$(date -u +%Y%m%dT%H%M%SZ).json"
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
config = json.loads(Path('neural_experiments/configs/runtime-v1.json').read_text())
config['primary_model_sha256'] = hashlib.sha256(
    Path(os.environ['H1_RUNTIME_MODEL']).read_bytes()
).hexdigest()
with Path(os.environ['H1_RUNTIME_CONFIG']).open('x') as stream:
    json.dump(config, stream, indent=2)
    stream.write('\n')
PY
```

Review and commit that new config before timing. Also commit any intentional
changes to the implementation/tests/protocol; the runner rejects dirty tracked
runtime implementation, tests, packaging and protocol paths. Keep the recipe and
implementation fixed between both final assessment stages.

```sh
git add -- "$H1_RUNTIME_CONFIG"
git commit -m "Pin reproduction runtime model identity"
python -m neural_decoder.runtime export --model "$H1_RUNTIME_MODEL" \
  --onnx "$H1_RUN/runtime-float32.onnx"
python -m neural_decoder.runtime profile --model "$H1_RUNTIME_MODEL" \
  --index "$H1_INDEX" --output "$H1_RUN/runtime-profile"
python -m neural_decoder.runtime validate --model "$H1_RUNTIME_MODEL" \
  --onnx "$H1_RUN/runtime-float32.onnx" --index "$H1_INDEX" \
  --config "$H1_RUNTIME_CONFIG" --output "$H1_RUN/runtime-validation"
python -m neural_decoder.runtime benchmark --model "$H1_RUNTIME_MODEL" \
  --onnx "$H1_RUN/runtime-float32.onnx" --index "$H1_INDEX" \
  --config "$H1_RUNTIME_CONFIG" --output "$H1_RUN/runtime"
python neural_experiments/audit_runtime.py --run "$H1_RUN/runtime" \
  --index "$H1_INDEX" --model "$H1_RUNTIME_MODEL" \
  --onnx "$H1_RUN/runtime-float32.onnx" \
  --output "$H1_RUN/runtime/independent-audit.json"
```

The benchmark includes full-trace parity before timing and retains per-bin
monotonic timestamps, predictions, every repeat/order, RSS, CPU, initialization,
reset costs and a separate kernel probe. Report service time, queue delay,
release-to-completion latency and deadline misses separately. The original NumPy
median fast p99 was 0.034333 ms; this is a measured machine/run result, not an
expected threshold for another machine or an official benchmark guarantee.

## New early and later freezes

The historical primary assessments are complete. Reproduction must create new
models and new freezes; the public freeze JSONs identify original private model
artifacts that are not distributed. Copying their digests does not authorize or
validate newly trained weights. Because the final outcomes are already public,
a reproduction is a fixed-protocol replication, not a newly untouched hypothesis
test.

[ASSESSMENT_DATA.md](ASSESSMENT_DATA.md) specifies the gateway. It checks all
current decoder Python files and the recipe against a real recorded Git commit,
model hashes, provider roles, metadata hashes and exact asset inventory. Later
source/config bytes must equal the early freeze's bytes. Documentation-only HEAD
changes are permitted when those frozen bytes still match. Keep all paths inside
the repository and assessment exports on a local, nonsymlink output directory.

Fit S0–S4 and freeze **before** releasing S5. The freeze command prints its SHA-256:

```sh
export H1_RECIPE="neural_experiments/configs/frozen-recipe-v1.json"
python -m neural_decoder.assessment fit --stage early --index "$H1_INDEX" \
  --recipe "$H1_RECIPE" --cache "$H1_CACHE" --output "$H1_RUN/early-models"
python -m neural_decoder.assessment freeze --run "$H1_RUN/early-models" \
  --recipe "$H1_RECIPE" --cache "$H1_CACHE" \
  --output "$H1_RUN/early-freeze.json" | tee "$H1_RUN/early-freeze.sha256"
export H1_EARLY_SHA="$(cat "$H1_RUN/early-freeze.sha256")"
```

Review the completed training manifest, frozen source/recipe/model identities and
S5 evaluation role. Independently record this exact digest and the release
decision before the next command. A hash proves identity, not authority or truthful
training provenance. This explicit export is the step that opens S5 sample arrays;
it does not use the ordinary development loader.

```sh
python - <<'PY'
import os
from pathlib import Path
from neural_decoder.assessment_data import export_assessment
run = Path(os.environ['H1_RUN'])
index = export_assessment(
    run / 'early-freeze.json', os.environ['H1_EARLY_SHA'],
    root=Path.cwd(), cache=Path(os.environ['H1_CACHE'])
)
(run / 'early-index-path.txt').write_text(str(index) + '\n')
PY
export H1_EARLY_INDEX="$(cat "$H1_RUN/early-index-path.txt")"
python -m neural_decoder.assessment evaluate --index "$H1_EARLY_INDEX" \
  --freeze "$H1_RUN/early-freeze.json" --sha256 "$H1_EARLY_SHA" \
  --cache "$H1_CACHE" --output "$H1_RUN/early-evaluation"
python neural_experiments/audit_assessment.py --run "$H1_RUN/early-evaluation" \
  --index "$H1_EARLY_INDEX" --freeze "$H1_RUN/early-freeze.json" \
  --sha256 "$H1_EARLY_SHA" --cache "$H1_CACHE" \
  --output "$H1_RUN/early-evaluation/independent-audit.json"
```

After that primary S5 evaluation, refit the **identical recipe** on S0–S5. The
later freeze references the early freeze and digest. S5 may enter this refit;
its score must not change hyperparameters, seeds, epochs, preprocessing or metrics.

```sh
python -m neural_decoder.assessment fit --stage later --index "$H1_INDEX" \
  --recipe "$H1_RECIPE" --cache "$H1_CACHE" --output "$H1_RUN/later-models" \
  --early-index "$H1_EARLY_INDEX" --early-freeze "$H1_RUN/early-freeze.json" \
  --early-sha256 "$H1_EARLY_SHA"
python -m neural_decoder.assessment freeze --run "$H1_RUN/later-models" \
  --recipe "$H1_RECIPE" --cache "$H1_CACHE" \
  --early-freeze "$H1_RUN/early-freeze.json" --early-sha256 "$H1_EARLY_SHA" \
  --output "$H1_RUN/later-freeze.json" | tee "$H1_RUN/later-freeze.sha256"
export H1_LATER_SHA="$(cat "$H1_RUN/later-freeze.sha256")"
```

Review and independently record the new later digest/release before exporting
S6–S12. The gateway creates a fresh digest-named directory under
`$H1_CACHE/prepared_assessment/`; it never overwrites an existing export.

```sh
python - <<'PY'
import os
from pathlib import Path
from neural_decoder.assessment_data import export_assessment
run = Path(os.environ['H1_RUN'])
index = export_assessment(
    run / 'later-freeze.json', os.environ['H1_LATER_SHA'],
    root=Path.cwd(), cache=Path(os.environ['H1_CACHE'])
)
(run / 'later-index-path.txt').write_text(str(index) + '\n')
PY
export H1_LATER_INDEX="$(cat "$H1_RUN/later-index-path.txt")"
python -m neural_decoder.assessment evaluate --index "$H1_LATER_INDEX" \
  --freeze "$H1_RUN/later-freeze.json" --sha256 "$H1_LATER_SHA" \
  --cache "$H1_CACHE" --output "$H1_RUN/later-evaluation"
python neural_experiments/audit_assessment.py --run "$H1_RUN/later-evaluation" \
  --index "$H1_LATER_INDEX" --freeze "$H1_RUN/later-freeze.json" \
  --sha256 "$H1_LATER_SHA" --cache "$H1_CACHE" \
  --output "$H1_RUN/later-evaluation/independent-audit-primary.json"
```

Each final model starts fresh at each provider file and predicts all bins before
scorer labels/masks are joined. Scores concatenate scored rows within each day,
compute variance-weighted R², and average days equally. The completed study's
independent audits checked 10 early and 70 later full prediction traces using
whole-file Torch or independent FIR arithmetic and sklearn scoring.

## Separate supervised calibration and private replay

Only after primary scoring, reproduce the prespecified intercept correction.
Budgets are scored labels **within each file's first 0/5/15/30 nominal seconds**,
not accumulated seconds of scored labels. All arms ingest preceding neural bins
identically and score the same suffix starting at bin 1500 (30 seconds). This is
an explicitly supervised secondary analysis; it does not alter primary results.

```sh
python -m neural_decoder.assessment calibration --index "$H1_LATER_INDEX" \
  --freeze "$H1_RUN/later-freeze.json" --sha256 "$H1_LATER_SHA" \
  --cache "$H1_CACHE" --run "$H1_RUN/later-evaluation" \
  --output "$H1_RUN/calibration"
python neural_experiments/audit_assessment.py --run "$H1_RUN/later-evaluation" \
  --index "$H1_LATER_INDEX" --freeze "$H1_RUN/later-freeze.json" \
  --sha256 "$H1_LATER_SHA" --cache "$H1_CACHE" \
  --calibration-run "$H1_RUN/calibration" \
  --output "$H1_RUN/later-evaluation/independent-audit-with-calibration.json"

python neural_experiments/replay_demo.py --index "$H1_LATER_INDEX" \
  --freeze "$H1_RUN/later-freeze.json" --sha256 "$H1_LATER_SHA" \
  --cache "$H1_CACHE" --primary-run "$H1_RUN/later-evaluation" \
  --model tcn-seed-0 --file S6_set_1 \
  --runtime-summary "$H1_RUN/runtime/summary.json" \
  --output "$H1_RUN/recorded-replay.html"
```

The standalone HTML validates hashes and displays saved predictions plus
scorer-side target comparisons for all seven outputs. Optional `--runtime-summary`
adds a separate static table of measured service statistics from a validated
aggregate; its development weights differ from the displayed final refit, and
browser playback performance is unmeasured. Play/pause/scrub controls
replay a recording; they do not execute live inference or control a device. The
HTML embeds sample data, must remain private, and is accepted only under ignored
`results/` or `work/`. Its receipt binds the selected data/model/freeze/prediction
and HTML hashes. Do not publish it with the code or aggregate reports.

## Compression and M2 extensions

Run these after the required H1 work and outside any other timing window. The
[compression protocol](COMPRESSION.md) defines the single dynamic-int8 setting,
quality budget and counterbalanced runtime experiment. Use the selected dropout
development run for both commands, keeping it separate from same-weight float32
parity. The audit checks the changed arithmetic as an approximation, not a
numerically identical implementation.

```sh
python neural_experiments/compression.py quality \
  --development-run "$H1_RUN/tcn-dropout" --index "$H1_INDEX" \
  --output "$H1_RUN/compression-quality"
python neural_experiments/audit_compression.py \
  --quality-run "$H1_RUN/compression-quality" \
  --development-run "$H1_RUN/tcn-dropout" --index "$H1_INDEX" \
  --output "$H1_RUN/compression-quality/independent-audit.json"
python neural_experiments/compression.py benchmark \
  --development-run "$H1_RUN/tcn-dropout" --index "$H1_INDEX" \
  --quality-run "$H1_RUN/compression-quality" \
  --output "$H1_RUN/compression-runtime"
python neural_experiments/audit_runtime.py --run "$H1_RUN/compression-runtime" \
  --index "$H1_INDEX" --model "$H1_RUN/tcn-dropout/seed-0/S4/model.npz" \
  --quality-run "$H1_RUN/compression-quality" \
  --output "$H1_RUN/compression-runtime/independent-audit.json"
```

The [M2 protocol and exact inventory](SECOND_TASK_PLAN.md) and
[completed M2 review](M2_REVIEW.md) describe seven minival NWBs, one fixed ridge
fit plus a constant comparator, and three future dates. This small check transfers
the implementation/fitting recipe, not H1 weights; sparse-training quality failed
and was retained without more tuning. M2 has distinct provider terms. Its runner
reuses the H1 upstream source cache, adds one bounded M2 cache, and requires the
explicit post-primary-work release flag:

```sh
python neural_experiments/transfer_m2.py \
  --config neural_experiments/configs/m2-minival-transfer-v1.json \
  --output "$H1_RUN/m2-transfer" --released-after-primary-work
python neural_experiments/audit_m2.py "$H1_RUN/m2-transfer"
```

Run the M2 audit once per fresh run; unlike the H1 audit CLIs it writes fixed
`independent-audit.json` and `independent-audit-source.py` names. Preserve earlier
receipts. Official EvalAI authentication, container validation and hidden-service
submission remain separate work; none of these commands submits an official score.

## Evidence to retain

Keep each run's completion/failure manifest, configuration/environment/source
snapshot, input/model/prediction hashes, full private traces and independent audit
receipts. Public material can contain code, protocols, provider metadata, freezes
and aggregate metrics; the private model/sample files referenced by their hashes
are intentionally absent. [SCORING.md](SCORING.md) explains aggregation and the
[notebook](NOTEBOOK.md) records decisions. The data and runtime documents retain
their dated pre-execution text as provenance; their current-state banners link to
the completed evidence. The original image experiment remains documented in
[the repository's CNN record](../CNN_EXPERIMENT.md).
