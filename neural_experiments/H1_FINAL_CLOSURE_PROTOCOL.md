# Final H1 deployment evidence: frozen protocol v1

Declared October 24, 2026, before final-model conversion measurements. The starting
revision is `d8085cc4c46ddaa1661eb507ae8e6fdd5a650927`. This is a bounded deployment
validation on previously exposed S6–S12, not a new generalization experiment.
The [configuration](configs/h1-final-closure-v1.json) pins all inputs and settings.
Historical reports, failed adaptations, weights, and the entire frozen Python
inventory under `src/neural_decoder/` remain unchanged. There is no refitting,
normalization change, quantization search, or early-stage secondary analysis.

## Identity and quality

Authenticate all three final later-stage models, both complete freeze digests,
assessment and development indices, historical evaluation manifest, all historical
prediction hashes, and byte-identical public/private scores. Use the existing
assessment loaders, with their historical source and containment checks intact.
Reconstruct whole-file float32 references before conversion. A historical mismatch
stops the study. Run seed 0, 1, and 2; seed 0 alone is fixed for timing, irrespective
of any outcome. Its quality is separate from the three-seed mean.

| Arm | Execution | Meaning |
|---|---|---|
| A | PyTorch float32 sliding window | Reference deployment |
| B | NumPy float32 incremental | Stateful scheduling of the same mapping |
| C | Original ONNX float32 | Explicit-history export |
| D | Rewritten ONNX float32 | Layout-only control |
| E | Rewritten ONNX dynamic int8 | Fixed arithmetic approximation |

Use ONNX 1.19.1, ORT 1.23.2, opset 17, legacy `dynamo=False` export. Preserve the
four exact constant-left matrix transposes and unchanged remaining operations.
Retain `quantization-input.onnx` as D. E uses per-channel QInt8 weights, dynamic
activation quantization, `reduce_range=False`, and exactly four `MatMulInteger`
operators; histories/interfaces remain float32. Inspect graphs independently.

All paths advance every bin, including unscored bins, and reset only at provider
file boundaries. Predictors receive no labels, masks, or trial markers. Preserve
zero missing latent history and training-derived transforms. Save complete traces
before joining targets and masks. Concatenate files by day before variance-weighted
R², retaining all seven output scores, negative values, and valid/invalid counts.
A–D must match whole-file references on every bin at `rtol=3e-5, atol=1e-7`, and
absolute day-score error must be at most `1e-5`. E must lose no more than `0.005`
absolute R² in any of the 21 seed/day cells. Report output-specific harm including
`tz` regardless of the weighted gate. This is an engineering approximation budget,
not a population noninferiority test. Independent auditing must reconstruct traces,
scores, transforms, gates, and provenance. Mutation fixtures must reject tampering.

## Timing and admission

Successful independently authenticated quality/equivalence gates are required per
arm before the campaign. Remove only unsupported or quality-ineligible arms, retain
alphabetical order, and use all cyclic rotations of that list. With five arms the
orders are ABCDE, BCDEA, CDEAB, DEABC, EABCD, in both fast and paced modes. No arm is
excluded for poor speed and no unfavorable repeat is replaced.

One benchmark owner runs one measurement process with no competing task-owned
training, numerical audit, export, or tests. Each arm/block uses a fresh subprocess.
Freeze tracked clean dependency sources (including new runners, auditors, protocol,
config, and helpers) before starting; retain their hashes and source snapshots.
Never edit dependencies between arms. Record CPU/OS/RAM, package/thread-pool versions,
power settings when readable, host load and background conditions. Use CPU only,
Torch/BLAS and ORT intra/inter-op threads 1, ORT sequential/full optimization and
spinning disabled. Reuse the installed environment and `work/runtime_deps`; no
native compilation, package synchronization, download, or new environment.

Every block warms 500 real bins outside measurement then resets. Fast blocks replay
all 11 complete chronological development files (112,488 bins). Paced blocks replay
the first 3,001 real bins of S0_set_1, releases spanning 60 seconds at 20 ms. Inputs
include final-model training days; timing is a systems workload, with separate
later-day quality evidence. Keep the slow first measured bin and every late bin.

Retain integer monotonic release, start, ready, complete, deadline, file, sequence,
and cycle identities. Deadline is release + 20,000,000 ns; frozen ready equals
complete because state update finishes before return. Service includes numeric
conversion, validation, transforms, inference and state work. Copying/logging the
returned prediction is outside service but contributes to wall time and queueing.
Fast releases equal processing starts. Paced queue = start − release, service =
complete − start, delivery = complete − release. Miss = complete > deadline;
backlog is floor(queue / period), not measured physical acquisition-queue occupancy.

Report numpy linear p50/p95/p99/max for each distribution, misses/count/rate,
maximum backlog, constructor and resets separately, graph/numeric-state sizes,
post-warmup RSS and process peaks, measured CPU/wall duration excluding audits.
The worker retains input arrays and libraries; RSS is not model-only memory.
Preserve absolute repeat values before A/B, C/D, D/E, C/E and E/B ratios. The project
service target is p99 <2 ms, with 20 ms delivery deadlines. There is no required
speedup, energy claim, or hard real-time guarantee. Counts-ready timing excludes
acquisition, bin accumulation, transport and physiological response. A 15-bin causal
receptive field describes past context, not future lookahead or measured response.

## Sustained replay and recovery

Run 90,000 measured bins (30 minutes at 50 Hz) for B. Also run E iff its conversion
quality and numerical lifecycle checks pass. Follow complete development files in
canonical order, cycling as needed; mark the last partial file. Reset at every real
file boundary, including cycle starts. Maintain one release-clock origin throughout,
so resets, writes and transitions remain visible in queue/delivery measurements.
Repeated recordings add no biological data. Use a fixed 4,096-record buffer backed
by preallocated durable logs and RSS sampling at 1 Hz. Report capacities, log bytes,
high-water/error/drop state, input-cache size and state sizes; sampled RSS alone
does not prove leak freedom. No growing per-bin Python lists or unbounded queue.

Checkpoint original float32 C and qualified E using the existing ONNX-only wrapper.
B receives reset/chunk/state-boundary checks without a new serialization system.
The experiment adapter validates supplied channel IDs against authenticated feature
metadata, with timestamp `sequence_id * 20,000,000 ns` from zero within each file.
Host clocks are separate. Reject wrong model/graph/precision/stream/next sequence,
channel count/order, nonfinite or negative input, duplicates, time gaps and corruption.
An explicit discontinuity or missing sequence marks the adapter unusable until an
explicit reset/new segment or validated restore. Arithmetic exceptions also make it
unusable. Rejected validation must not partially change numeric state. Validate restore
through a fresh candidate before replacing the active wrapper; bind channel/cadence
metadata to its sidecar. Unannounced channel permutations cannot be detected numerically.

Retain a prior complete checkpoint; reject a truncated later file. Create-new writes
are not atomic publication and do not establish interrupted-write durability. Numeric
resume does not establish exactly-once external delivery. Use composite run/cycle/file
stream IDs. Measure bare C/E versus wrapper/adapter complete-step overhead separately
for 3,001 real prefix bins, forward/reverse order, with 500 warmup and reset. Fault
runs inject one 60 ms stall before bin 10 in 31 paced bins, retaining backlog/misses
and processing every observation. Fault results never enter clean timing statistics.

## Attempts, audits and release

Create a new directory for every attempt, with start/final execution status, exact
commands, source/config/input/environment bindings and errors. Distinguish artifact
completion, independent verification, float32 equivalence, compression quality and
runtime target status. Preserve failures even if artifact audits later succeed.
Repair an operational/reporting defect under a new source identity and new attempt;
never retry solely for favorable performance. Runtime auditors recompute integer-clock
metrics, schedules/order/reset/warmup and measured predictions: A–D against independent
final seed-0 float32 references, E against independent E replay. Int8 self-consistency
is distinct from its approximation gate.

Raw recordings, derived samples, weights, per-bin predictions, timing and real-data
replay remain local in ignored `results/`, `data/` and `work/`. Publish only aggregate
scores/timing/resources, source/config identities, audit receipts, protocols, code and
synthetic tests. Bound new outputs to 2 GiB; retain historical evidence. Hosted tests
must use synthetic data only. No private upload, paid compute or external publication
is authorized by this protocol. Required local checks use existing Python binaries:

```sh
export PYTHONPATH="$PWD:$PWD/src:$PWD/work/runtime_deps"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/ruff check src/neural_decoder tests/neural_decoder neural_experiments
.venv/bin/ruff format --check src/neural_decoder tests/neural_decoder neural_experiments
.venv/bin/python -m pytest -q tests/neural_decoder
```

New command entry points are inventory/quality (`h1_final_closure.py`), independent
quality audit (`audit_h1_final_closure.py`), benchmark/soak
(`h1_final_closure_runtime.py`), fault-check (`h1_final_closure_recovery.py`), and
runtime audit (`audit_h1_final_closure_runtime.py`). Each accepts explicit create-new
output locations. Exact executed commands and source identities belong in the report.

Stop performance work for historical identity/reproduction failure. An E quality
failure ends quantization work and preserves qualified float32 comparisons. Unsupported
paths are explicit exclusions. The final report must pair seed-0 quality/timing,
retain all-three-seed results and failures, and distinguish model bytes from RSS,
layout from approximation, and CPU service from delivery. Optional early-stage checks,
second hardware, visual polish and new science are omitted before required evidence.
