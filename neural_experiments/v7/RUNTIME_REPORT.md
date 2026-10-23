# V7B runtime: bounded source replay and refitting

The selected decoder completed two 60-second, 50 Hz paced replays with **zero
20 ms deadline misses across 6,000 predictions**. Release-to-completion p99 was
5.564 and 5.547 ms. Ten accumulating-label refits, including model export and
reload, took a median 7.567 ms and a maximum 9.233 ms on this machine, well within
the evaluation's assumed two-second refit allowance for this workload.

These are source-data numerical and compute measurements. They do not measure
fresh decoding accuracy, hard realtime guarantees, neural acquisition latency,
or closed-loop BCI performance. The original measurement attempt is retained;
there was no rerun to select a more favorable timing result.

## Workload and scope

The workload was selected before timing outcomes: the greatest numeric exposed
T2 day, day 250, and the last complete `periodic240` fit. It uses 240 purchased
label seconds in five intervals, with 9,493 eligible training rows. The last
labels end at 540 seconds and the model is deployed at 542 seconds in the
scientific source replay. Selection used metadata rather than decoding scores.

The benchmark used the exact serialized model from the independently audited
source run. Each replay restored the same causal state behavior, including
500 warmup bins and acquisition-block resets. Three fast replays each processed
all 30,000 bins in the 600-second prefix. Two paced replays each processed
3,000 bins, releasing one step every 20 ms for 60 seconds.

The final model contains labels acquired later than most of the replayed
features. Therefore, the full-prefix outputs are numerical parity checks and
compute workloads, not causal efficacy predictions. Predictions from 542 seconds
onward were additionally compared with the actual source deployment trace.

Per-step service timing includes input validation, feature transforms, causal
filtering, normalization, the linear head, output projection and any reset at
that step. It begins with features already available in memory. Acquisition,
upstream feature extraction, transport and file loading are outside this timing.
Release-to-completion timing additionally includes scheduler delay.

Refit timing includes selecting the purchased-label mask, fixed first-30-second
normalization, ridge fitting, serialization and model reload. It assumes causal
representations and labels are already cached in memory. Building that cache,
loading inputs and obtaining labels are outside the refit timing. The cached
representation occupied 46,080,000 bytes in this workload.

## Measurements

| Replay | Predictions | Service p99 (ms) | Release-to-completion p99 (ms) | 20 ms deadline misses |
|---|---:|---:|---:|---:|
| Fast 1 | 30,000 | 0.01879 | 0.01883 | 0 |
| Fast 2 | 30,000 | 0.01946 | 0.01950 | 0 |
| Fast 3 | 30,000 | 0.01721 | 0.01729 | 0 |
| Paced 1 | 3,000 | 1.450 | 5.564 | 0 |
| Paced 2 | 3,000 | 1.291 | 5.547 | 0 |

The fast replay tests arithmetic throughput. Its immediate releases do not
reproduce a periodic input stream, so the paced rows are the relevant finite
scheduling observations. Paced scheduler-delay p99 was 4.914 and 4.943 ms;
maximum measured service time was 4.323 and 3.813 ms. Each paced block lasted
approximately 60.005 seconds. Measured process CPU time was 0.957 and 1.030
seconds, about 1.60% and 1.72% of each block's wall time.

For the ten refits, total time had p50 **7.567 ms**, p95 **9.114 ms**, and maximum
**9.233 ms**. None exceeded two seconds. Every restored refit had exactly the
same numeric model arrays as the selected source model. This result assesses
the two-second allowance on one bounded workload; it does not establish a
worst-case bound for larger models, different hardware or simultaneous workloads.

Streaming versus batch predictions differed by at most **9.16 × 10⁻¹⁵**. After
the source deployment boundary, predictions matched the source scientific trace
exactly. The decoder's persistent numeric filter state was 3,072 bytes and its
normalizer/head arrays occupied 6,160 bytes. These are component sizes, not total
application memory. Process peak RSS reached 472,023,040 bytes and included
loaded inputs, cached representations and parity workspaces.

The model's causal filter uses a 200 ms time constant. Its steady-state mean
sample age is approximately 190 ms and its 95% step response takes 30 bins
(600 ms). These describe temporal smoothing and should not be confused with
the much shorter compute service time.

## Environment and retained evidence

The run used macOS 26.2 on ARM64, Python 3.12.13, NumPy 2.5.2 and SciPy 1.18.1.
The process requested a one-thread numerical-library limit and set
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. The thread-pool inspection
returned an empty list, so that inspection does not independently establish
the threading behavior of every underlying numerical library.

During the measurement attempt, a separate synthetic diagnostic test ran for
approximately 0.79 seconds and a background download was active. This ambient
load was reported during execution and is recorded here without changing the
original manifest, timestamps or source snapshot. Its effect on individual
timing samples was not isolated. No favorable-attempt selection was performed.

The private run directory is `results/neural_v7/continual-runtime/`. It retains
the immutable manifest, source snapshots, five timestamp/prediction artifacts
and ten exported refit models. Each prediction block retains raw release,
start and completion clocks, enabling independent reconstruction of service,
queue, completion and deadline statistics. Refit receipts retain elapsed fit,
export/reload and total durations; their arithmetic and reported quantiles can
be independently checked, but raw refit start/solve/end clocks were not saved.

The [benchmark implementation](runtime_continual.py) and
[focused synthetic tests](../../tests/neural_decoder/test_v7_runtime_continual.py)
are separate from the frozen scientific producer. The
[continual protocol](CONTINUAL_PROTOCOL.md) defines efficacy and label costs;
runtime observations do not alter its thresholds or decisions. Sample-level
traces, models and prepared inputs remain local.
