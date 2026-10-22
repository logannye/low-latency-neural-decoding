# Reproducing V2

V2 asks two separate questions: how much supervised recalibration improves an
existing participant's decoder, and whether a temporal representation learned
from other people improves a new person's decoder at a fixed label budget.
Read the [locked protocol](PROTOCOL.md), [H1 accuracy study](ACCURACY.md), and
[independent audit specification](AUDIT.md) before interpreting a score.

Run commands from the repository root using Python 3.12 and installed CPU wheels.
No Rust compilation, GPU service, or external pretrained weights are required.

```sh
python -m pip install --only-binary=:all: -e '.[dev,neural,benchmark]'
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m pytest -q tests/neural_decoder
```

## H1 development experiments

Follow the [original acquisition guide](../README.md#acquire-and-prepare-the-pinned-h1-data)
to obtain the pinned public H1 calibration files. Reuse a validated existing
prepared index; do not rewrite data during a run. Choose fresh output directories
for every execution. Existing receipts and unsuccessful attempts are evidence.

```sh
python -m neural_experiments.v2.accuracy \
  --index data/neural_h1/prepared/index.json \
  --plan neural_experiments/v2/accuracy_plan.json \
  --output results/neural-v2/accuracy-reproduction
python -m neural_experiments.v2.audit_accuracy \
  --run results/neural-v2/accuracy-reproduction \
  --index data/neural_h1/prepared/index.json \
  --output results/neural-v2/accuracy-reproduction-audit.json
python -m neural_experiments.v2.compose_head \
  --run results/neural-v2/accuracy-reproduction \
  --index data/neural_h1/prepared/index.json \
  --output results/neural-v2/composed-head-reproduction
```

The last command folds the already-fitted 30-second residual correction into
the existing linear head. It performs no additional fitting. The source-unit
correction is transformed algebraically into the model's standardized output
coordinates. The exported decoder retains 23,367 parameters and its original
streaming operations. Only the suffix after the calibration prefix supports a
quality claim; replaying that fixed artifact over an entire file measures runtime.

## Nine-participant acquisition and preparation

The [BrainGate release](https://doi.org/10.5061/dryad.x0k6djj1h) provides the source
data under CC0. The importer pins Dryad version 462702 and the nine decoding
archives, then retains the first six regular decoding-session MAT members in
archive order, including short and zero-go sessions.
It streams bounded prefixes rather than downloading all 79 GB. Published archive
hashes are recorded as provider metadata; local member hashes establish the
identity of the files actually consumed. A full-archive checksum is not claimed.

```sh
python -m neural_experiments.v2.data metadata --cache data/neural_v2
python -m neural_experiments.v2.data acquire --cache data/neural_v2 --sessions 6
python -m neural_experiments.v2.data prepare --cache data/neural_v2
python -m neural_experiments.v2.data validate --cache data/neural_v2
```

If Dryad requires browser verification, obtain the archives through its ordinary
download flow. Acquisition accepts `--archive-directory` for local provider
archives, or `--url-file` for a private participant-to-URL JSON map containing
provider redirects observed in that flow. Expiring signed URLs belong only in an
ignored private file and must not be committed. Preserve failed acquisition
receipts. Do not infer a missing participant from a transient download failure.
Defaults bound each compressed stream to 4 GiB and the cache to 20 GiB.

The importer sums adjacent 10 ms counts within each block. Targets are the
second bin's normalized target-minus-cursor direction. Both source bins must
be eligible movement-attempt bins for the combined row to score. Non-scored
observations remain causal history; block gaps reset state and odd trailing
bins are accounted for. No whole-session target-person centering is used.
Prepared features and scorer-only targets are separate files, with strict
schema, chronology, shape, and hash validation.

## Participant-held-out training and independent scoring

```sh
python -m neural_experiments.v2.transfer \
  --index data/neural_v2/index.json \
  --plan neural_experiments/v2/transfer_plan.json \
  --output results/neural-v2/transfer-reproduction
python -m neural_experiments.v2.audit \
  --run results/neural-v2/transfer-reproduction \
  --index data/neural_v2/index.json \
  --output results/neural-v2/transfer-reproduction-audit.json
```

The runner performs 27 source fits: nine excluded people times three seeds.
Each source model has participant-specific input adapters and readouts, with
one shared width-32 residual temporal core. Channel positions have no assumed
cross-person neuron identity. Five dilated causal layers provide 63 bins of
history, with an identity path through each layer. The held-out person receives
a new adapter and head. The frozen-core, fine-tuned-core, and scratch arms have
the same architecture and target training schedule. Causal ridge is an explicit
low-capacity comparison. All four arms share the same prefix and effective loss
rows, including the common per-block 62-bin exclusion.

Only other people's last selected sessions choose the source checkpoint.
Target-person statistics use the calibration prefix, and later days receive
no further updates. First-session scoring begins after 300 seconds for every
feasible budget. Short recordings produce unavailable cases, not replacement
sessions. Budgets count observed 20 ms bins, not verified elapsed wall time
across acquisition gaps. The primary comparison is frozen core versus scratch
at 120 seconds on future days; larger descriptive scores cannot replace it.

The auditor independently reads retained arrays and uses scikit-learn to
reconstruct scores. It checks all planned cases, input normalizers, frozen-core
identity, participant exclusion, common masks, seed/day/person aggregation,
bootstrap intervals, and exact participant sign-flip statistics. It does not
retrain the models or validate upstream voltage-processing causality.

## Exploratory freezing control

After observing the primary results, we added one random-frozen-core arm to
separate learned source weights from the regularization effect of freezing.
The [separate plan](random_core_plan.json) declares that prior exposure and pins
the original run, source, index, audit, and public summary. It is post-hoc
analysis on the same cohort, not a second confirmatory test. It uses all nine
people, all three original seeds, the 120-second prefix, and the original
40-epoch schedule. No search or model selection is added.

```sh
python -m neural_experiments.v2.random_core_control run \
  --index data/neural_v2/index.json \
  --plan neural_experiments/v2/random_core_plan.json \
  --reference-run results/neural-v2/transfer-v1 \
  --reference-public neural_experiments/results/v2-transfer.json \
  --reference-audit results/neural-v2/transfer-audit-v1.json \
  --output results/neural-v2/random-core-reproduction
python -m neural_experiments.v2.random_core_control audit \
  --run results/neural-v2/random-core-reproduction \
  --output results/neural-v2/random-core-reproduction-audit.json \
  --public-output neural_experiments/results/random-core-reproduction.json
```

These commands assume the exact retained original reference artifacts. For a
fresh primary reproduction, make a separate control-plan copy with that run's
actual reference hashes, retaining its recipe and exposure declarations. Do not
assign the published run's acceptance to new artifacts. The control runner and
its audit helper share a source file; numerical scoring uses the separately
implemented scikit-learn reference. This is narrower than an independently
implemented training pipeline or decoder replay.

## Same-artifact latency

Stop training and data preparation before timing. Each command checks complete
trace parity, runs three fast repeats with alternating order, then two 60-second paced
blocks per implementation. Pacing releases one observation every 20 ms. No
target arrays or scoring masks enter the timing workload.

```sh
python -m neural_experiments.v2.runtime --kind h1 \
  --model results/neural-v2/composed-head-reproduction/model.npz \
  --features data/neural_h1/prepared/features/S4_set_1.npz \
  --output results/neural-v2/h1-runtime-reproduction
```

The timing audit requires artifacts to resolve inside the checkout. If a shared
cache is a symlink outside it, copy the selected feature file into an ignored
local run directory and verify identical SHA256 before passing that copy to
`--features`. No refitting or data transformation is involved.

After the runtime command's built-in timestamp reconstruction, run the separate
auditor and aggregate-only export:

```sh
python -m neural_experiments.v2.audit_runtime \
  --run results/neural-v2/h1-runtime-reproduction \
  --output results/neural-v2/h1-runtime-reproduction-audit.json
python -m neural_experiments.v2.public runtime \
  --run results/neural-v2/h1-runtime-reproduction \
  --audit results/neural-v2/h1-runtime-reproduction-audit.json \
  --output neural_experiments/results/h1-runtime-reproduction.json
```

For transfer, use `--kind transfer`, the frozen-core 120-second seed-0 model
for the first eligible participant in plan order, and that person's earliest
feature file from the index. This rule does not select the fastest or most
accurate participant. The NumPy implementation retains bounded activation
history; PyTorch recomputes the observed receptive-field window using identical
serialized weights. Full-step service time includes input validation,
normalization, inference, state updates, and recording-boundary reset. Report
scheduling delay and release-to-completion separately. Timestamp reconstruction
runs automatically. These are offline feature-ready measurements on a general
purpose operating system, not a hard real-time or acquisition-latency guarantee.

## Artifacts and figures

Each execution records its plan, consumed source snapshots, environment, data
index, normalizers, model identities, and complete prediction traces. These
remain under ignored `data/` and `results/`. Public JSON contains aggregate
metrics, case dispositions, and hashes, never model arrays or sample traces.
Public figure generation reads those aggregate files only:

```sh
python -m neural_experiments.v2.figures \
  --accuracy neural_experiments/results/v2-accuracy.json \
  --outputdir neural_experiments/figures
```

Use the optional `--transfer` and `--runtime` arguments with completed public
summaries. Synthetic tests require no neural-data download. Re-executing training
on a different numerical stack can change floating-point results and artifact
hashes; new receipts must identify that environment rather than inherit this
run's acceptance.
