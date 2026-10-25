# Final H1 deployment evidence

**Status: all prescribed measurements completed and independently verified. The overall provisional 2 ms service-p99 goal failed; both 30-minute soaks retained deadline misses. Local software checks passed. Code and aggregate evidence are published in draft PR #11; hosted status is linked below.** See the [frozen protocol](H1_FINAL_CLOSURE_PROTOCOL.md), [historical H1 report](FINAL_REPORT.md), and [reviewer guide](REVIEWER_GUIDE.md).

The exact final 23,367-parameter H1 models reproduced their historical later-day results, and all four float32 execution paths passed full-file numerical and day-score equivalence. The fixed int8 conversion passed the prespecified approximation budget in all 21 seed/day cells: its worst variance-weighted R² change was −0.001715406, within the −0.005 limit. The predetermined seed-0 exported graph shrank from 102,884 to 39,641 bytes, a 61.47% reduction.

The five-path runtime control changes the explanation of the speedup. Rewriting the float32 matrix layout reduced median fast-replay p99 from 0.057541 to 0.025250 ms, a 2.28× ratio. Adding dynamic int8 reduced that median only to 0.024542 ms, a further 1.029× ratio. The smaller representation therefore passed a useful quality/storage tradeoff, while most of the original-ONNX-to-int8 fast p99 difference was already present in the layout-only control. These observations describe the configured graphs, runtime and host; they do not isolate a particular CPU kernel or establish a universal benefit from lower precision.

## Main comparison

Quality uses the same three final models, each evaluated separately on all seven S6–S12 days. The seed-0 reference mean R² was 0.177086334, and the average of the three reference scores was 0.177583285. The latter is an average of seed scores, not an ensemble. The timed model is always seed 0. The main table’s p99 columns are the **median of five per-run p99 values**, not a pooled percentile. All times are milliseconds. Graph bytes are serialized ONNX files; numeric state excludes weights, interpreter/runtime objects, scratch allocations and the loaded replay cache.

| Execution path / precision | Seed-0 SHA-256 prefix | Seed-0 later-day R² | All-seed mean ΔR² | Worst seed/day ΔR² | Graph bytes | Numeric state bytes | Fast service p99, seed 0 | Paced service p99, seed 0 | Paced delivery p99, seed 0 | Paced misses / updates | Soak status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A: PyTorch sliding window float32 | model `95f6a0b3131e` | 0.177086333 | +0.000000000 | -0.000000002 | n/a; NPZ 98,266 | 14,464 | 0.111250 | 1.381041 | 5.828250 | 0 / 15,005 | not prescribed |
| B: NumPy incremental float32 | model `95f6a0b3131e` | 0.177086343 | -0.000000002 | -0.000000054 | n/a; NPZ 98,266 | 14,464 | 0.033422 | 0.596292 | 5.154250 | 0 / 15,005 | 90,000 bins; 1 miss |
| C: Original ONNX float32 | graph `38f27f22c6d5` | 0.177086357 | +0.000000001 | -0.000000051 | 102,884 | 2,944 | 0.057541 | 0.938250 | 5.393584 | 0 / 15,005 | not prescribed |
| D: Rewritten ONNX float32 | graph `a420643c3c86` | 0.177086345 | +0.000000000 | -0.000000064 | 105,840 | 2,944 | 0.025250 | 0.578042 | 5.200084 | 0 / 15,005 | not prescribed |
| E: Rewritten ONNX dynamic int8 | graph `7dbedc6cbf3d` | 0.177615377 | +0.000322623 | -0.001715406 | 39,641 | 2,944 | 0.024542 | 0.632375 | 5.197708 | 0 / 15,005 | 90,000 bins; 5 misses |

All graph rows derive from the same seed-0 model; full digests are listed below and bound in the aggregate receipts. E uses signed per-channel int8 weights and dynamic activation quantization; its external inputs, outputs and persistent histories remain float32.

A–D retained strict full-trace parity, including startup and unscored bins, at rtol 3e−5 and atol 1e−7 in original output units. Their largest absolute full-trace differences across all seeds/files were 7.45e−9, 1.77e−8, 1.58e−8 and 1.91e−8 respectively. Every absolute day-score difference was below 1e−5. E is explicitly approximate; its largest absolute prediction difference was 0.000737920 in the original output units. Its small positive average score change is an observed arithmetic perturbation, not evidence of a trained-model improvement or a reason to change the frozen gate.

The ONNX paths each retain 2,944 numeric state bytes. The current A/B engine allocates 14,464 persistent numeric-array bytes, including both window and ring arrays even when a mode leaves some unused. In the paced subprocesses, peak observed RSS ranged from roughly 349–371 million bytes across paths, with overlapping ranges among C, D and E. Every process also retained 80,091,456 input-cache bytes. The graph reduction does not establish lower process memory or energy consumption.

## Every repeat and the failed runtime goal

The study completed all 50 scheduled blocks: five cyclic orders for five arms, separately in fast and 50 Hz paced modes. Each fast block replayed all 112,488 bins from the 11 development files with a reset at each provider-file boundary. Each paced block replayed the same first 3,001 bins from S0_set_1, covering 60 seconds between scheduled first and last release. All decoders warmed for 500 bins outside measurement and reset before the first measured stream. The first measured bin was retained.

The S0–S4 timing recordings were part of the final models' training data; later-day quality is measured separately on S6–S12. Cyclic orders place each arm in every position but do not balance every possible predecessor pair.

The project goal of p99 complete-step service below 2 ms **failed for the overall campaign**: B reached 2.258333 ms in paced repeat 4, and C reached 2.780750 ms in paced repeat 4. All other short-campaign p99 values were below 2 ms. No retry removed these failures.

| Arm | Mode | Repeat 0 p99 | Repeat 1 p99 | Repeat 2 p99 | Repeat 3 p99 | Repeat 4 p99 | Maximum single-bin service | Deadline exceedances / updates |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | fast | 0.093917 | 0.192333 | 0.221089 | 0.111250 | 0.093172 | 6141.404750 | 2 / 562,440 |
| A | paced | 1.919500 | 1.406917 | 1.381041 | 0.425250 | 0.414084 | 5.157333 | 0 / 15,005 |
| B | fast | 0.032791 | 0.033422 | 0.034416 | 0.033875 | 0.031584 | 0.924959 | 0 / 562,440 |
| B | paced | 0.772833 | 0.479458 | 0.596292 | 0.247833 | 2.258333 | 4.580833 | 0 / 15,005 |
| C | fast | 0.057541 | 0.067417 | 0.061625 | 0.053125 | 0.057125 | 132.188792 | 1 / 562,440 |
| C | paced | 0.938250 | 0.998000 | 0.774333 | 0.285000 | 2.780750 | 6.541958 | 0 / 15,005 |
| D | fast | 0.023797 | 0.027000 | 0.049958 | 0.025250 | 0.023833 | 151.265250 | 1 / 562,440 |
| D | paced | 0.578042 | 0.657333 | 0.703959 | 0.191375 | 0.336625 | 2.551500 | 0 / 15,005 |
| E | fast | 0.024584 | 0.022917 | 0.045458 | 0.024125 | 0.024542 | 0.326125 | 0 / 562,440 |
| E | paced | 0.632375 | 0.699791 | 0.741542 | 0.172500 | 0.297958 | 4.450667 | 0 / 15,005 |

All five arms had zero deadline misses in their paced runs, each totaling 15,005 measured updates. That finite observation is compatible with the failed 2 ms service goal because a deadline is measured against the 20 ms scheduled release interval. It is not a hard real-time guarantee.

Fast replay retained substantial outliers: A had a maximum step of 6,141.404750 ms and two 20 ms deadline exceedances; C reached 132.188792 ms and D reached 151.265250 ms, each with one exceedance. These were recorded during the clean fast campaign and remain part of its results. The logs do not establish their cause. No inference that a decoder kernel itself ran continuously for that entire elapsed interval is warranted; the measurement is wall-clock service and includes host descheduling. User background workload was uncontrolled, and a process inventory was unavailable.

The fresh-process design isolates each arm/block's constructor, runtime state and observed memory. Torch intra-op and BLAS were limited to one thread; Torch inter-op was neither explicitly set nor recorded. ONNX Runtime explicitly limited intra-op and inter-op pools to one, using CPU execution, sequential mode, identical graph optimization and disabled spinning. The observed host was macOS 26.2 ARM64 with 16 logical and 16 physical cores and 128 GiB RAM. The harness did not identify the processor model or a usable power-mode value, so this report does not relabel the host as an M4 Max from historical context. ONNX was 1.19.1 and ONNX Runtime 1.23.2.

## What the controls resolve

A→B reduced median fast p99 by 3.33× using the same final weights, while retaining the full causal computation. C→D reduced it by 2.28×, D→E by 1.029×, and C→E by 2.34×. E's fast median was 1.36× lower than the incremental NumPy incumbent's median. E was slightly slower than D in two of the five fast repeats, which further limits a quantization-only speed claim.

Paced ordering was different. Median p99 service was 0.578042 ms for D, 0.596292 ms for B and 0.632375 ms for E. Their median p99 delivery times were 5.200084, 5.154250 and 5.197708 ms respectively. Median per-run p99 queueing was approximately 4.8–4.9 ms across all arms. These separate percentiles are not additive decompositions of a single bin; they nevertheless show that scheduled delivery was strongly affected by wakeup and host scheduling delay. A lower fast service percentile did not translate into a clear scheduled-delivery advantage.

Percentiles use linear interpolation (`numpy.percentile(method="linear")`) on intervals derived from retained integer monotonic timestamps. Original source-bin timestamps are preserved separately. All p50/p95/p99/max values, per-run misses and rates, calculated backlog, constructor/reset costs, CPU/wall time and memory fields are retained in the aggregate evidence.

The measured boundary begins when precomputed counts are available. Service includes the selected numeric engine's host conversion, checks, preprocessing, inference and state update. Prediction-ready and completion timestamps coincide. Copying predictions into the durable log and synchronous log flushing occur outside service but remain visible to wall time and subsequent paced delivery. Acquisition, spike detection, bin accumulation, transport and human feedback are outside this boundary. The 15-bin causal receptive field describes past context and does not imply either 300 ms of future lookahead or a measured intent-response delay.

## Quality cells and output-specific limits

The fixed int8 gate is no seed/day loss larger than 0.005 absolute variance-weighted R². It is an engineering approximation budget, not a population noninferiority test. The 21 cells are:

| Day | Seed 0 ΔR² | Seed 1 ΔR² | Seed 2 ΔR² |
|---|---:|---:|---:|
| S6 | +0.000095549 | +0.000435989 | -0.000300645 |
| S7 | +0.000199034 | +0.000131612 | -0.000470110 |
| S8 | +0.001031731 | +0.001762983 | +0.001407954 |
| S9 | -0.000563728 | -0.000603824 | -0.000412201 |
| S10 | +0.002018139 | +0.001070658 | -0.000184729 |
| S11 | +0.002276186 | +0.001922035 | +0.001436876 |
| S12 | -0.001353610 | -0.001409410 | -0.001715406 |

A passing weighted score does not imply that every output changed by less than 0.005. The largest output-specific loss was tx on seed 1/S9, decreasing by 0.008747663. The historically weak tz output remained poor: on seed 1/S9 its R² changed from −0.717955279 to −0.719347091. The table retains the worst delta for each original source-labeled output, with the corresponding absolute scores. These values were not used to retune conversion or revise the original gate.

| Output | Worst ΔR² | Seed / day | Float32 reference R² | Int8 R² |
|---|---:|---|---:|---:|
| tx | -0.008747663 | 1 / S9 | -0.386037894 | -0.394785558 |
| ty | -0.003017696 | 2 / S10 | 0.149869685 | 0.146851989 |
| tz | -0.001391812 | 1 / S9 | -0.717955279 | -0.719347091 |
| rx | -0.003365958 | 2 / S12 | 0.136654073 | 0.133288115 |
| g1 | -0.001819458 | 0 / S12 | 0.344807112 | 0.342987654 |
| g2 | -0.001764943 | 0 / S12 | 0.319912252 | 0.318147310 |
| g3 | -0.003077525 | 1 / S12 | 0.341403348 | 0.338325823 |

The assessment contains 14 files, 34,490 total bins and 33,613 scored bins, about 689.8 recorded seconds from one participant. Models were trained on S0–S5 and frozen before the historical S6–S12 assessment. Those later days are now exposed; this work validates fixed conversion on previously assessed recordings. Multiple seeds, repeated timing blocks and software soak loops do not add independent participants, recording days or new generalization evidence. No new calibration, normalization, adaptation, training or ensemble was introduced. The result does not establish T2 recovery or cross-person transfer.

## Lifecycle and separate fault evidence

B passed chunk-versus-contiguous replay, state equality and file-start reset checks. C and E passed exact checkpoint continuation and 17 explicit lifecycle checks, including wrong model/graph/precision/stream identity, wrong next sequence, supplied channel-order mismatch, shape and finite/nonnegative input validation, missing input, timestamp discontinuity, truncated checkpoint rejection, and arithmetic-failure poisoning until reset. The retained complete checkpoint survived the truncated later-file test. Channel checks validate supplied metadata; a same-shaped numeric vector cannot reveal an unannounced channel permutation.

Independent direct ONNX execution reconstructed every stored lifecycle/overhead trace and the checkpoint suffix, and separately recomputed the integer timestamp checks. This verifies deterministic numerical continuation. It does not establish exactly-once delivery of external outputs, an atomic checkpoint publication protocol or interrupted-write durability.

Wrapper/metadata validation overhead was measured separately from the bare-engine campaign. C's bare p99 values were 0.050292 and 0.052500 ms, versus 0.058291 and 0.059584 ms with the cadence adapter. E's bare values were 0.024125 and 0.023709 ms, versus 0.032291 and 0.033208 ms with the adapter. These diagnostics include validation inside the complete-step boundary and cannot be substituted for the separately measured paced campaign.

Each 31-bin injected-stall test requested a 60 ms pause before sequence 10. Both C and E retained all inputs, reported three deadline misses and a maximum calculated backlog of three bins, then drained the backlog. Those fault traces are separate from the clean runtime distributions. Backlog here is calculated from the host release schedule; there is no independently measured acquisition queue.

## Sustained replay and resource bounds

B and E each completed the prescribed 90,000 bins at 50 Hz, with every prediction and timestamp independently verified. Each run used one host-clock origin across ten file starts: nine complete development files followed by the first 1,485 bins of S4_set_1, beginning at global bin 88,515 and ending at file sequence 1,484. Neither run wrapped the 112,488-bin input inventory. The nominal duration was 1,800 seconds, with 1,799.98 seconds between first and last scheduled releases. These are reused recordings, not new biological observations or preserved real acquisition gaps.

| Soak metric | B: NumPy float32 | E: ONNX int8 |
|---|---:|---:|
| Updates | 90,000 | 90,000 |
| Service p50 / p95 / p99 / max, ms | 0.088750 / 0.349333 / 0.450417 / 4.017750 | 0.283458 / 0.561500 / 0.731335 / 22.356917 |
| Queue p50 / p95 / p99 / max, ms | 4.505916 / 7.883502 / 9.467420 / 29.727166 | 3.979416 / 7.393102 / 9.097103 / 46.297500 |
| Delivery p50 / p95 / p99 / max, ms | 4.624417 / 8.111670 / 9.610334 / 29.867375 | 4.141666 / 7.720551 / 9.449300 / 46.621916 |
| Missed deadlines / updates | 1 / 90,000 | 5 / 90,000 |
| Miss rate | 0.001111% | 0.005556% |
| Maximum calculated backlog, bins | 1 | 2 |
| Wall duration, seconds | 1,799.983815 | 1,799.990564 |
| Process CPU time, seconds | 13.655844 | 29.590335 |
| Constructor time, ms | 0.674792 | 15.501917 |
| Reset count / total duration, µs | 10 / 88.335 | 10 / 146.001 |
| Reset duration range, µs | 3.708–16.542 | 5.417–35.000 |
| Post-warmup RSS, bytes | 351,567,872 | 370,081,792 |
| Sampled RSS min–max, bytes | 351,584,256–360,218,624 | 370,130,944–378,880,000 |
| After-replay RSS / lifetime peak, bytes | 360,611,840 | 379,273,216 |
| RSS samples / maximum sample interval, seconds | 1,778 / 1.026512 | 1,779 / 1.029304 |
| Synchronous flush count / total time, ms | 22 / 47.150003 | 22 / 50.837293 |

Both soaks passed the provisional service-p99 target, but neither was free of missed deadlines. Int8 retained a 22.356917 ms service outlier and a 46.621916 ms maximum delivery delay. NumPy's single miss occurred despite every service interval remaining below 20 ms, illustrating why service and scheduled delivery need separate checks. No late result was discarded or replaced. One sequential B/E pair does not establish a causal difference in long-run miss rates, CPU cost or host scheduling behavior.

Each run retained 80,091,456 input-cache bytes, a fixed 409,600-byte staging buffer (4,096 records × 100 bytes), and a preallocated 9,000,000-byte mapped record payload. Headers, JSON logs and source snapshots are additional disk storage. The high-water mark was 4,096 records, every one of the 90,000 records was written, and zero records were dropped. The writer was synchronous with no asynchronous queue; backpressure remained visible to subsequent scheduled delivery. Decoder numeric state remained 14,464 bytes for B and 2,944 bytes for E.

RSS sampling was nominally once per second; median observed intervals were 1.017709 seconds for B and 1.016977 for E. The sampled minima and maxima were also the first and last samples. Post-warmup RSS was captured before logger allocation, so subsequent growth includes mapped log pages. Lifetime peak includes initialization, input loading and warmup; discrete RSS samples can miss brief peaks. These observations and fixed capacities establish bounded retained state/log allocations for this workload, not general absence of memory leaks.

CPU accounting includes replay, resets, logging and RSS sampling; it excludes initialization, warmup, later summary calculations and auditing. It is not a power measurement. The retained worker status records place B at 01:31:56–02:01:56 UTC and E at 02:01:58–02:31:58 UTC on October 25, 2026. The reused environment helper's `environment.started_utc` field is the environment snapshot time taken after replay; use the run/worker start and final status records for execution boundaries.

The short campaign also retains constructor and reset costs across all ten processes per arm (five fast and five paced), with sixty file resets per arm:

| Arm | Constructor range, ms | Reset range, µs |
|---|---:|---:|
| A | 1.070000–2.058750 | 3.083–14.542 |
| B | 0.466917–0.612584 | 2.042–12.791 |
| C | 12.508084–17.096792 | 3.542–9.875 |
| D | 12.765500–18.120292 | 2.708–14.708 |
| E | 12.778333–33.407458 | 2.542–13.375 |

## Mechanism and implementation decision log

These entries connect the frozen comparison to the observed evidence; they are not claims that the measurements uniquely identify an internal runtime mechanism.

| Decision | Observed issue or cost | Proposed mechanism and smallest falsifying check | Result and disposition |
|---|---|---|---|
| Authenticate and reconstruct before export | Earlier runtime/compression evidence used developmental weights | Bind model, freeze, index and historical prediction bytes; independently reconstruct whole-file traces and scores | All three exact final models reproduced; keep the artifact-bound study |
| Retain incremental NumPy as the incumbent | Sliding windows recompute prior hidden activations | Cache causal layer histories; any startup/full-trace mismatch falsifies equivalence | B passed strict parity and reduced median fast p99 versus A by 3.33×; keep B as a qualified float32 path |
| Time rewritten float32 separately | Existing complete conversion confounded matrix layout with quantization | Change only four constant-left matrix products and exact weight transposes, then check graph and full traces | D passed; C→D accounts for most median fast improvement. Keep the control and attribute the observed effect to the configured layout path |
| Retain the fixed int8 candidate without search | Smaller weights can add dynamic activation-quantization work | Require four MatMulInteger operators, independent exact int8 replay and all 21 fixed score-budget cells | E passed and reduced graph bytes by 61.47%; incremental fast improvement over D was small and paced delivery was similar. Keep E as an optional smaller representation |
| Keep float32 as the default | The compression gate does not settle every output or timing objective | Retain per-output losses, incumbent comparisons and all timing repeats | tx degraded more than 0.005 in one cell, B/C missed the provisional paced service goal, and no delivery advantage was established. Do not turn gate success into a blanket deployment recommendation |
| Validate stream metadata outside the bare-engine claim | Numeric state alone cannot identify channel order or a missing input | Exact stream/sequence/cadence/identity checks plus poisoned-state, reset and checkpoint tests | Qualified C/E lifecycle and measured adapter overhead; no exactly-once or atomic-write claim |
| Preserve wall-clock outliers | Good p99 can coexist with long isolated pauses | Retain every raw timestamp and recompute summary/miss counts independently | Four fast deadline exceedances across A/C/D retained; overall provisional service goal failed through two paced blocks. No timing reruns replaced them |

## Application wording

> Built a 23,367-parameter causal neural decoder that improved mean later-day R² from 0.034 to 0.178 across seven recording days from one participant, beating selected ridge in all 21 seed/day comparisons. Converted all three final models to a 61.5% smaller ONNX int8 graph with worst seed/day variance-weighted R² loss of 0.00172, using a float32 layout control to identify the main source of fast-replay improvement. Profiled the predetermined seed-0 int8 model at 50 Hz in offline replay, pairing later-day R² 0.1776 with 0.731 ms service p99 and 9.449 ms delivery p99 over 90,000 updates, retaining five missed deadlines and verifying deterministic checkpoint continuation.

This three-sentence claim uses the int8 soak's actual percentiles and miss count. Supporting evidence retains the failed overall short-campaign service goal, all outliers, output-specific losses and the one-participant, nonofficial-FALCON boundary. Quality and timing come from different identified recordings using the same exact weights; this is neither fresh generalization nor live control.

## Evidence and provenance

The quality, short-runtime, soak and lifecycle execution source was committed at `94d545758a10b18128b8c318e950111c0997f7b9`. The quality auditor passed the complete three-model conversion evidence; the runtime auditor passed all 50 short blocks and both soaks, including every measured prediction. Audit success authenticates execution and scoring evidence while preserving the failed runtime goal.

Private run roots:

- `results/neural-h1/h1-final-closure-quality-v1/` and `h1-final-closure-quality-audit-v1.json`.
- `results/neural-h1/h1-final-closure-runtime-v1/` and `h1-final-closure-runtime-audit-v1.json`.
- `results/neural-h1/h1-final-closure-recovery-v1/` including its retained `validation.json`.
- `results/neural-h1/h1-final-closure-soak-v1/` and `h1-final-closure-soak-audit-v1.json`.

The seed-0 model SHA-256 is `95f6a0b3131e1971f146298e23c95a42f7da28467d8acc24f37f916d9dc049ec`. Graph SHA-256 identities are C `38f27f22c6d5faea8d8d22f5397fbafd9706e1e3c7ea360762b0973f9bce699d`, D `a420643c3c86c1b20cdcfc7f07690db43e1da85c9f89c993cdd2d5c3d40cc0e5`, and E `7dbedc6cbf3dc853079eec9e4b1fb89d19ed4de942439ea26c0ee467fc8f4ea2`.

The [aggregate tables](results/h1-final-closure-v1/tables.md), [complete quality scores](results/h1-final-closure-v1/quality.json), [all short timing/resource summaries](results/h1-final-closure-v1/runtime.json), [soak summaries](results/h1-final-closure-v1/soak.json), [lifecycle results](results/h1-final-closure-v1/lifecycle.json), and [identity bindings](results/h1-final-closure-v1/bindings.json) accompany this report. Independent receipts cover [quality](results/h1-final-closure-v1/quality-audit.json), [all short blocks](results/h1-final-closure-v1/runtime-audit.json), [both soaks](results/h1-final-closure-v1/soak-audit.json), and [lifecycle trace validation](results/h1-final-closure-lifecycle-audit-v1.json). Consult the [historical quality protocol](PROTOCOL.md) for the original assessment.

The failed inventory-v1 attempt and successful inventory-v2 correction remain in the attempt log: the first adapter incorrectly rejected an existing shared development-cache symlink; the correction reused its pinned digest and historical authenticated cache loader without weakening frozen model/assessment containment. No historical file was overwritten, and no unfavorable timing sample was replaced. Raw H1 inputs, models, per-bin traces, checkpoints and replay logs remain local in ignored storage.

## Reproduction commands

Use the already-installed environment and authenticated local artifacts. The
following are the actual run names for this study; reproduce under fresh names.
The runners refuse to overwrite an existing attempt. Do not download, refit or
reprepare arrays to replace a failed identity check.

```sh
export PYTHONPATH="$PWD:$PWD/src:$PWD/work/runtime_deps"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python neural_experiments/h1_final_closure.py inventory \
  --config neural_experiments/configs/h1-final-closure-v1.json \
  --output results/neural-h1/h1-final-closure-inventory-v2
.venv/bin/python neural_experiments/h1_final_closure.py quality \
  --config neural_experiments/configs/h1-final-closure-v1.json \
  --output results/neural-h1/h1-final-closure-quality-v1
.venv/bin/python neural_experiments/audit_h1_final_closure.py \
  --run results/neural-h1/h1-final-closure-quality-v1 \
  --output results/neural-h1/h1-final-closure-quality-audit-v1.json
.venv/bin/python neural_experiments/h1_final_closure_recovery.py fault-check \
  --config neural_experiments/configs/h1-final-closure-v1.json \
  --quality-run results/neural-h1/h1-final-closure-quality-v1 \
  --quality-audit results/neural-h1/h1-final-closure-quality-audit-v1.json \
  --output results/neural-h1/h1-final-closure-recovery-v1
.venv/bin/python neural_experiments/h1_final_closure_runtime.py benchmark \
  --config neural_experiments/configs/h1-final-closure-v1.json \
  --quality-run results/neural-h1/h1-final-closure-quality-v1 \
  --quality-audit results/neural-h1/h1-final-closure-quality-audit-v1.json \
  --output results/neural-h1/h1-final-closure-runtime-v1
.venv/bin/python neural_experiments/audit_h1_final_closure_runtime.py \
  --run results/neural-h1/h1-final-closure-runtime-v1 \
  --output results/neural-h1/h1-final-closure-runtime-audit-v1.json
.venv/bin/python neural_experiments/h1_final_closure_runtime.py soak \
  --config neural_experiments/configs/h1-final-closure-v1.json \
  --quality-run results/neural-h1/h1-final-closure-quality-v1 \
  --quality-audit results/neural-h1/h1-final-closure-quality-audit-v1.json \
  --lifecycle results/neural-h1/h1-final-closure-recovery-v1/manifest.json \
  --output results/neural-h1/h1-final-closure-soak-v1
.venv/bin/python neural_experiments/audit_h1_final_closure_runtime.py \
  --run results/neural-h1/h1-final-closure-soak-v1 \
  --output results/neural-h1/h1-final-closure-soak-audit-v1.json
```

After all three audit receipts pass, create the aggregate-only report export:

```sh
.venv/bin/python neural_experiments/h1_final_closure_report.py \
  --quality results/neural-h1/h1-final-closure-quality-v1 \
  --quality-audit results/neural-h1/h1-final-closure-quality-audit-v1.json \
  --runtime results/neural-h1/h1-final-closure-runtime-v1 \
  --runtime-audit results/neural-h1/h1-final-closure-runtime-audit-v1.json \
  --soak results/neural-h1/h1-final-closure-soak-v1 \
  --soak-audit results/neural-h1/h1-final-closure-soak-audit-v1.json \
  --lifecycle results/neural-h1/h1-final-closure-recovery-v1/manifest.json \
  --output neural_experiments/results/h1-final-closure-v1
```

The lifecycle audit receipt is a byte-identical, hash-only copy of the retained
validation record. Copy it once to a fresh public destination after export:

```sh
.venv/bin/python - <<'PYCOPY'
from pathlib import Path
source = Path("results/neural-h1/h1-final-closure-recovery-v1/validation.json")
target = Path("neural_experiments/results/h1-final-closure-lifecycle-audit-v1.json")
with target.open("xb") as output:
    output.write(source.read_bytes())
PYCOPY
```

## Attempt log and validation status

| Attempt | Execution | Evidence result |
|---|---|---|
| Inventory v1 | Failed; retained | New adapter rejected the pre-existing shared development-cache symlink |
| Inventory v2 | Complete | Exact development cache hash and established validator accepted; historical containment checks preserved |
| Quality v1 | Complete | Historical reproduction, all four float32 equivalence gates and all 21 int8 cells passed independent audit |
| Recovery v1 | Complete | B/C/E lifecycle checks and separate raw-trace verification passed |
| Runtime v1 | Complete | All 50 blocks independently verified; overall provisional 2 ms service-p99 goal failed |
| Soak v1 | Complete | Both 90,000-bin traces and clocks passed independent audit; B retained 1 miss, E retained 5 |
| Aggregate export v1 | Complete | Authenticated receipts exported; raw sample arrays, models and timing traces excluded |

The protocol/configuration were committed at `94700b8` before new final-model
measurements; implementation at `94d5457` preceded quality, recovery and timing.
Task-owned private evidence occupies approximately 319 MiB on disk, below the
frozen 2 GiB output cap; useful failures and raw evidence were retained. No
final-model training, dependency installation, native compilation or private
artifact upload was performed. Hosted software checks are tracked in
[draft PR #11](https://github.com/logannye/low-latency-neural-decoding/pull/11/checks).
The first published head, `d4bb8a897aeee9a210d2d7d76880dc1d25670d54`, passed
[Neural decoding Python](https://github.com/logannye/low-latency-neural-decoding/actions/runs/40170440802)
and [Tests](https://github.com/logannye/low-latency-neural-decoding/actions/runs/40170440817).
These validate Linux software contracts and synthetic workflows; private H1
quality and macOS timing evidence are not rerun in CI. Later documentation-only
revisions have their own check records on the PR.
Final local validation on the execution code at `94d5457` passed **1,361 tests with no skips** in 35.11 seconds. The 52 warnings are legacy Torch ONNX-export deprecations. Lint passed and all 332 scoped Python files passed formatting. `git diff --check` passed; the complete diff from the starting revision contains no change under `src/neural_decoder/`. The local test log is retained at `results/neural-h1/h1-final-closure-validation-v1/pytest.log`.

The starting revision `d8085cc4c46ddaa1661eb507ae8e6fdd5a650927` has successful
[Tests](https://github.com/logannye/low-latency-neural-decoding/actions/runs/40006323051)
and [Neural decoding Python](https://github.com/logannye/low-latency-neural-decoding/actions/runs/40006323163)
workflows, refreshed during this study. Those receipts cover the starting revision,
not the new implementation. The authorized publication includes code, protocols,
aggregates and audit receipts; recordings, weights and per-bin traces remain local.
The PR check list identifies hosted outcomes for each published revision.

Final local validation commands (with the environment above):

```sh
NEURAL_H1_INDEX=data/neural_h1/prepared/index.json .venv/bin/python -m pytest -q tests/neural_decoder
.venv/bin/ruff check src/neural_decoder tests/neural_decoder neural_experiments
.venv/bin/ruff format --check src/neural_decoder tests/neural_decoder neural_experiments
git diff --check
git diff d8085cc4c46ddaa1661eb507ae8e6fdd5a650927 -- src/neural_decoder
```
