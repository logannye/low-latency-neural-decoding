# Frozen same-weight runtime comparison

This is a prespecified G4 experiment, not a measured speed claim. Root owns the
single measurement process. Commit the implementation, tests, this protocol and
`configs/runtime-v1.json` before running final timing. Preserve failed runs.

Use the preselected seed 0 development fold C model, evaluated on S4 during G2.
Do not choose the best seed. The identical float32 parameters and affine
transforms feed three implementations: existing eager PyTorch sliding window,
existing NumPy cached layer rings, and ONNX Runtime's optimized CPU graph with
explicit layer histories. This experiment has frozen inference and no adaptive
post-prediction work. Quantization, adaptation and model selection are separate.

First retain a 256-bin cProfile diagnostic of the two existing implementations.
Its instrumentation is useful for call structure/hotspots, not latency claims.
The systems hypothesis is to avoid repeated hidden-activation computation and
Python/kernel dispatch by expressing one causal step as an optimized graph.
That graph receives the new count vector plus histories of shapes `(2,176)`,
`(4,32)` and `(8,32)`, and returns the prediction and updated histories. It uses
the same oldest/middle/current convolution taps and zero missing latent history
as whole-file convolution. It does not invent pre-file biased activations.

Export uses PyTorch's explicit legacy exporter, `dynamo=False`, opset17. Install
prebuilt `onnx` and `onnxruntime` wheels; no native compilation or exporter
compiler is invoked. ONNX Runtime uses only `CPUExecutionProvider`, sequential
execution, full graph optimizations, one intra-op and one inter-op thread, and
disabled thread spinning. Torch/BLAS thread limits are also one. Record versions,
the source revision, artifact/configuration/input hashes and actual thread pools.
Graph export is a separate startup operation; each execution block records model
constructor time (including ORT session graph preparation), reset time and state
storage. Optional `CheckpointedONNXStream` snapshots store all three histories,
format version, numeric-model and exact ONNX-graph identities, precision,
stream/file identity, last consumed sequence/timestamp and next sequence ID.
Restore validates every field before mutation and requires the caller's expected
next sequence, preventing a resumed input from being skipped or replayed.
This optional metadata/checkpoint wrapper is tested separately and is not inside
the numeric-engine timing comparison. Its overhead is not covered by speed claims.

Before timing, compare every full trace from all 11 authorized development files
against whole-file PyTorch forward, including startup/unscored bins, using
`rtol=3e-5, atol=1e-7` in target units. Per-day variance-weighted R² differences
must also stay within `1e-5`, a numerical consistency gate. A compression-quality
budget such as 0.005 cannot excuse a same-weight arithmetic error. Save every
prediction trace and check measured-run predictions again. Mixed training and
development files here provide a runtime workload, not fresh biological evidence.
Never read sealed-day features or targets for this experiment.

Every implementation/block consumes a fixed 500-bin warm-up outside measured
timing, then resets at each measured file boundary. Fast replay uses all 11
development files in chronological file order, three repetitions, with Latin
rotation of implementation order. Paced replay uses the first chronological
file and 20 ms releases, two 60-second blocks per implementation in forward and
reverse order. This totals at least 120 seconds per implementation; the middle
implementation remains in the middle in these two orders. All 3,001 bins of a
60-second paced block are real observations; no repeated recording is claimed
as independent biological evidence.

For every input retain file identity, sequence ID, scheduled release, processing
start, prediction ready, step complete and next-bin deadline as integer monotonic
nanoseconds. A frozen step completes state updates before returning its output,
so prediction-ready and step-complete timestamps are equal. The service boundary
includes numeric conversion, preprocessing, inference, state replacement and
validation; copying/logging the returned prediction is outside service time.
Host logging and file-reset overhead remains visible in paced queue delay and
total wall/CPU time. Missed deadlines compare full completion with the next bin's
scheduled release, not merely a 20 ms service threshold. Backlog is queued whole
bin periods at processing start. No late bin is dropped, accelerated or replaced.

Report p50/p95/p99/max service and release-to-prediction/completion latency,
queue delays, missed deadlines, backlog, measured duration, initialization/reset,
parameter bytes, state-array bytes and process RSS before/after each block.
RSS is process memory, not exclusive decoder memory or an allocation count. CPU
percent is process CPU divided by wall time, including harness activity. Report
each repeat/order and absolute times before ratios. Do not infer power savings.
Retain a separate first-layer microbenchmark comparing three strided GEMVs with
one contiguous flattened GEMV, using identical weights/history and arithmetic
parity. Run 10,000 iterations in each of two reversed orders, after 500 warm-ups.
These Python-invoked kernel samples explain dispatch/layout costs; they do not
replace full decoder service measurements. Graph export wall time is reported
separately from each ORT session's constructor/optimization time.
The provisional p99 service goal is 2 ms within a 20 ms bin period; passing it is
an engineering replay result, not a clinical or Neuralink requirement.

Commands from repository root, with the installed runtime-wheel directory on
`PYTHONPATH` if applicable:

```sh
python -m neural_decoder.runtime profile --model MODEL.npz --index INDEX.json --output NEW_PROFILE
python -m neural_decoder.runtime export --model MODEL.npz --onnx NEW_MODEL.onnx
python -m neural_decoder.runtime validate --model MODEL.npz --onnx MODEL.onnx --index INDEX.json --output NEW_PARITY
python -m neural_decoder.runtime benchmark --model MODEL.npz --onnx MODEL.onnx --index INDEX.json --output NEW_RUN
```

Only benchmark invokes the final campaign and requires committed source. Runtime
results establish counts-ready processing, excluding bin accumulation, electrodes,
spike detection, wireless transport and actual neural-to-action latency.
