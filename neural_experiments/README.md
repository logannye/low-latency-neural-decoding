# Neural decoding across recording days

This separate study asks whether a small causal movement decoder remains useful
across days with no new-day labeled calibration. It currently completes the data
and baseline gate and the first compact temporal-model comparison. The original
image-classification experiment remains unchanged.

**Development result:** a 23,367-parameter TCN averages R² **0.1607** across three
seeds and three forward-day folds, versus **0.0526** for the selected causal ridge
baseline. S2 remains difficult and seed-dependent. These development scores were
used to guide the study; they are neither untouched final results nor official
FALCON scores. See [REPORT.md](REPORT.md) for every seed/day, limitations, validation,
and the next hypothesis. S5-S12 remain sealed to model development.

## Reproduce

Use Python3.12 and prebuilt scientific dependencies. The current local environment
is `.venv`. A fresh environment can install `python -m pip install --only-binary=:all:
-e '.[dev,neural,benchmark]'`; this project introduces no compiled extension or Rust build.
Run these commands from the repository root. H1 download requires accepting its
[provider data-use terms](data/README.md); recordings and prepared arrays remain
ignored and local. Use one cache. Reacquisition uses exact manifest asset UUIDs,
so a newer mutable DANDI draft cannot silently change the dataset.

```sh
export PYTHONPATH="$PWD/src"
.venv/bin/python -m neural_decoder.data acquire --cache data/neural_h1 \
  --manifest neural_experiments/data/h1_manifest.json
.venv/bin/python -m neural_decoder.data prepare --cache data/neural_h1
NEURAL_H1_INDEX=data/neural_h1/prepared/index.json .venv/bin/pytest -q
.venv/bin/neural-decode baseline --index data/neural_h1/prepared/index.json \
  --output results/neural-h1/ridge-reproduction
.venv/bin/neural-decode temporal --index data/neural_h1/prepared/index.json \
  --output results/neural-h1/tcn-pilot-reproduction --pilot
.venv/bin/neural-decode temporal --index data/neural_h1/prepared/index.json \
  --output results/neural-h1/tcn-reproduction
```

Run directories must be new. Preserve failures and use another name for retries.
Preparation exports S0-S4 only; S5-S12 undergo schema metadata inspection only.
No calibration/minival sample array becomes independent evidence by changing its
filename. Current preparation may rewrite the same prepared arrays; retain the
index/array hashes and do not reprepare an existing experiment mid-run.

Replay one feature-only recording, with a reset at the start:

```sh
.venv/bin/neural-decode replay --kind tcn \
  --model results/neural-h1/tcn-reproduction/seed-0/S2/model.npz \
  --features data/neural_h1/prepared/features/S2_set_1.npz \
  --output results/neural-h1/replay-S2-set1.npy
```

`replay` loads no target/mask/trial file. This is recorded-data replay, not live
neural control. Reference sliding-window and incremental TCN implementations have
trace-parity checks. Runtime speed, p99 service time, deadlines and sustained load
have not yet been measured. The batch-one H1 adapter is tested synthetically;
an official challenge package/container/submission has not been exercised.

## Evidence

- [Protocol](PROTOCOL.md) and [notebook](NOTEBOOK.md).
- [Data/schema/terms](data/README.md) and exact asset/source/split manifests.
- [Scoring contract](SCORING.md), [ridge independent review](BASELINE_REVIEW.md),
  [TCN design](TCN.md), and [TCN independent review](TCN_REVIEW.md).
- `results/neural-h1/ridge-dev-v1` and `tcn-dev-v1`: local source snapshots,
  configuration/environment/input identities, model artifacts, predictions,
  scores, losses, completion manifests, and independent audit receipts.
- `neural_experiments/audit.py` and `audit_temporal.py` independently recompute
  saved evidence. Raw/sample-level results remain local under the data-use terms.

The new neural CI workflow runs synthetic correctness and train-export-replay.
The badges on the repository README link to actual hosted run status.
The original image workflow remains intact.
