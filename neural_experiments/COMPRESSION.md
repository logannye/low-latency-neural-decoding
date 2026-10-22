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
  --development-run results/neural-h1/tcn-dropout-dev-v1 \
  --index data/neural_h1/prepared/index.json \
  --output results/neural-h1/compression-quality-v1
PYTHONPATH=src python neural_experiments/compression.py benchmark \
  --development-run results/neural-h1/tcn-dropout-dev-v1 \
  --quality-run results/neural-h1/compression-quality-v1 \
  --index data/neural_h1/prepared/index.json \
  --output results/neural-h1/compression-runtime-v1
```

The completed experiment uses the selected dropout development recipe in both
commands. These commands do not grant access to sealed evaluation arrays.

## Completed results

The fixed conversion passed the nine-model quality rule. Mean day R² changed by
**+0.001731**, and the worst seed/day change was **−0.000617** (seed 2/S3), within
the −0.005 limit. Small increases do not establish that quantization generally
improves decoding: these are the same three development days used to assess the
chosen dropout recipe, from one participant, with no compressed sealed-day test.

| Seed | Day | Float32 R² | Dynamic-int8 R² | Difference |
|---|---|---:|---:|---:|
| 0 | S2 | -0.063407 | -0.060479 | +0.002928 |
| 0 | S3 | 0.311225 | 0.312931 | +0.001706 |
| 0 | S4 | 0.315891 | 0.316152 | +0.000262 |
| 1 | S2 | -0.095459 | -0.090822 | +0.004636 |
| 1 | S3 | 0.235801 | 0.236420 | +0.000619 |
| 1 | S4 | 0.278058 | 0.277538 | -0.000520 |
| 2 | S2 | 0.013870 | 0.020277 | +0.006407 |
| 2 | S3 | 0.237938 | 0.237321 | -0.000617 |
| 2 | S4 | 0.315457 | 0.315611 | +0.000154 |

Every graph shrank from **102,884 to 39,641 bytes**, a **61.47% reduction**.
Input/output tensors and the 2,944-byte persistent history remain float32. The
conversion first rewrites four constant-left products `W @ v` as `v @ W.T` to
support the runtime's dynamic quantizer. Independent inspection verified exact
weight transposes and unchanged remaining operations. Full traces from the
rewritten float32 graphs matched their originals within the locked numerical
tolerance; the largest absolute difference was **1.86e-08**. Int8 saved traces
were reproduced exactly by a separate explicit-state ONNX replay, and sklearn
independently reproduced all day and output scores. This verifies the conversion
and accounting without treating quantization as same-weight numerical parity.

Timing used source revision `8290ed6c836cb5f3aa156dcc3232480da93e4017`, CPU
ONNX Runtime 1.23.2 on the same M4 Max/macOS 26.2 environment, and the preselected
dropout seed 0/S4 model. Its NPZ SHA256 is
`14b6cd3a620dab84e15e942cb307c510114d290a6eb1808e4899980007c9a5a5`.
These weights differ from the original TCN used in the primary same-weight
runtime study, so the two campaigns are not a matched comparison across models.
Each fast repeat contains 112,488 bins. Each paced block contains 3,001 bins whose
scheduled releases span 60 seconds. Both modes use float32 then int8 in repeat 0,
and int8 then float32 in repeat 1; all eight blocks passed independent raw-stamp,
hash, workload and summary auditing.

| Mode | Repeat | Precision | Service p99, ms | Queue p99, ms | Release-to-completion p99, ms | Deadline misses |
|---|---:|---|---:|---:|---:|---:|
| Fast | 0 | Float32 | 0.054709 | 0 | 0.054709 | 0 |
| Fast | 0 | Int8 | 0.025292 | 0 | 0.025292 | 0 |
| Fast | 1 | Int8 | 0.025333 | 0 | 0.025333 | 0 |
| Fast | 1 | Float32 | 0.054339 | 0 | 0.054339 | 0 |
| Paced | 0 | Float32 | 0.859958 | 4.875291 | 5.406833 | 0 |
| Paced | 0 | Int8 | 0.989292 | 4.871958 | 5.316792 | 0 |
| Paced | 1 | Int8 | 0.899709 | 4.885167 | 5.329625 | 0 |
| Paced | 1 | Float32 | 1.098208 | 4.864500 | 5.404458 | 0 |

The converted graph's median fast-repeat p99 is about **2.15× lower**. Paced
service-time rankings reverse between repeats, while release-to-completion p99
remains dominated by roughly 4.9 ms of scheduling delay. No compression arm
missed a deadline in 6,002 paced bins each; this is an observation over 120 seconds
per arm, not a hard real-time guarantee. The untimed transposed float32 graph is
not a separate runtime control, so the fast improvement belongs to the complete
conversion, including graph rewriting, rather than an isolated int8 arithmetic
effect. The default deployment precision remains float32; int8 is an explicit
optional approximation and does not replace the primary assessed models.

Constructor/reset times, process RSS, and CPU accounting are retained in the
aggregate records. CPU totals include a second validation replay and file
persistence. RSS includes loaded data, libraries and allocations retained from
previous blocks. Neither is an isolated deployment memory or energy measurement.
The public [quality results](results/compression-quality.json) and
[runtime results](results/compression-runtime.json) contain aggregate scores,
counts, sizes and artifact hashes. The corresponding
[quality audit](results/compression-quality-audit.json) and
[runtime audit](results/compression-runtime-audit.json) retain independent checks;
sample arrays, model weights and raw timing traces remain local.
