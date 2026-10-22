# Separate compression experiment

This optional experiment follows the required frozen primary evaluation. It
converts all four explicit-state TCN matrix products to per-channel signed int8
weights with dynamic activation quantization in ONNX Runtime. The setting is
fixed once: no reduced range, no tuning, and no selection of a favorable seed.
The graph's float32 inputs and outputs do not make its internal arithmetic
float32. Metadata explicitly identifies dynamic int8, and the default deployment
loader rejects it unless that precision is requested.

Use all three seeds and three forward-day development folds of the selected TCN
recipe. Each model is compared with its own saved float32 predictions on the
same full files and masks. Retain per-day/output scores, full traces, model/graph
hashes and sizes. The prespecified quality gate requires no seed/day to lose more
than 0.005 absolute R². This is an approximation tradeoff, not same-weight parity.

The runtime comparison uses the preselected seed0/S4 model. Both arms use CPU
ONNX Runtime, one thread, batch one, the same observations, reset per file, and
500 explicitly declared warmup bins. Fast replay covers all 11 development files;
paced replay covers 60 seconds per block at 20 ms/bin. Execute each pair in both
orders. Save every timing sample and recompute p50/p95/p99/max, queue delay,
release-to-output/completion time, and missed deadlines. Python allocations and
host scheduling remain part of the experiment. No power or acquisition claim.

Even if the quality gate passes, deployment remains float32 unless measured
runtime provides a useful improvement. A smaller graph alone is not evidence of
lower full-step latency. Any unsupported kernel or runtime failure remains in
its own fresh failure directory; no unreported fallback is permitted.

```sh
PYTHONPATH=src python neural_experiments/compression.py quality \
  --development-run results/neural-h1/tcn-dev-v1 \
  --index data/neural_h1/prepared/index.json \
  --output results/neural-h1/compression-quality-v1
PYTHONPATH=src python neural_experiments/compression.py benchmark \
  --development-run results/neural-h1/tcn-dev-v1 \
  --quality-run results/neural-h1/compression-quality-v1 \
  --index data/neural_h1/prepared/index.json \
  --output results/neural-h1/compression-runtime-v1
```

If the augmentation decision selects another development run, use that run in
both commands. These commands do not grant access to sealed evaluation arrays.
