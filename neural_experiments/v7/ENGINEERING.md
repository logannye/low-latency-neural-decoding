# V7 engineering: fast inference and bounded calibration state

The source replay supports two concrete engineering results. The exported
decoder met every 20 ms deadline in two finite 60-second paced tests. A separate
sufficient-statistics implementation reproduced all five accumulating-label
fits while retaining **302,624 bytes of calibration state**, independent of the
number of acquired labels. Neither result changes the frozen scientific
evaluation or establishes fresh decoding efficacy.

## Runtime of the exported decoder

The benchmark used the latest numeric exposed T2 day, day 250, and its final
`periodic240` fit. This representative was chosen by metadata before timing
outcomes. The final model used labels through 540 seconds and was deployed at
542 seconds in the source evaluation.

| Measurement | Result |
|---|---:|
| Paced predictions | 6,000 across two 60-second runs |
| 20 ms deadline misses | 0 |
| Release-to-completion p99, runs 1 / 2 | 5.564 / 5.547 ms |
| Service p99, runs 1 / 2 | 1.450 / 1.291 ms |
| Ten cached-representation refits, median / maximum | 7.567 / 9.233 ms |
| Streaming versus batch maximum prediction error | 9.16 × 10⁻¹⁵ |
| Persistent numeric filter state | 3,072 bytes |
| Numeric normalizer and head arrays | 6,160 bytes |

The independent audit reconstructed all five timestamp blocks, checked ten
refit exports, and verified the source-model binding. Detailed measurements,
scope and the operational annotation are in the
[runtime report](RUNTIME_REPORT.md) and
[audited public aggregate](../results/v7-runtime.json).

Three full 600-second fast replays tested arithmetic and numerical parity.
Because the final model contains later-acquired labels, its earlier replay
predictions are not causal accuracy evidence. The two paced replays measure
finite scheduling behavior after features are available; acquisition, upstream
feature extraction and transport are outside that timing. A 200 ms causal
filter time constant also remains distinct from compute time.

The original attempt is retained. A 0.79-second synthetic diagnostic test and a
background download overlapped it; their individual timing effects were not
isolated. Raw prediction clocks were saved. Refit receipts saved elapsed
fit/export/total durations, enabling independent arithmetic and quantile checks,
but not reconstruction from raw refit start/end clocks. An empty thread-pool
inspection cannot independently establish numerical-library threading, despite
the requested single-thread limits. These qualifications are preserved in the
public artifact rather than removed by a favorable rerun.

## Retaining information without retaining every label row

The scientific decoder fixes normalization to the first 30 seconds, clips
normalized features to ±8, and solves mean squared error with ridge penalty 1
on feature coefficients and an unpenalized intercept. With a fixed feature
map, the ridge solution depends on a small set of sufficient statistics rather
than the individual labeled rows.

For feature dimension \(d\) and two output coordinates, the new accumulator
retains a count \(n\), means \(\bar{x},\bar{y}\), centered Gram matrix
\(S_{xx}\), and centered cross-product \(S_{xy}\). It obtains coefficients and
intercept from

\[
B=(S_{xx}/n+I)^{-1}(S_{xy}/n),\qquad
b=\bar{y}-\bar{x}B.
\]

This is algebraically the same mean-loss ridge objective as the augmented
batch solve. Centered block merging includes the between-block mean correction,
avoiding the cancellation caused by subtracting large uncentered sums.

The implementation processes at most 64 causal feature rows per update. It
accepts only purchased, valid past labels, excludes the same 62-bin reset burn,
and rejects out-of-order or duplicate observations. Labels outside the selected
mask may be nonfinite without entering arithmetic. Counts below the original
50-label floor retain the same zero-model fallback. The label policy,
normalizer, penalty, output projection and deployment times remain unchanged.

A separate runner streamed the declared source session once, accumulated
the original paid intervals, and solved at each of the five original cutoffs.
The original exported models were references, not fitting inputs. Each fitted
model was exported and reloaded before comparison.

| Label cutoff (s) | Eligible purchased rows | Maximum coefficient error versus original export |
|---|---:|---:|
| 150 | 4,743 | 1.19 × 10⁻¹⁵ |
| 240 | 5,960 | 8.88 × 10⁻¹⁶ |
| 360 | 7,057 | 1.19 × 10⁻¹⁵ |
| 480 | 8,274 | 1.39 × 10⁻¹⁵ |
| 540 | 9,493 | 1.47 × 10⁻¹⁵ |

The [independent audit receipt](../results/v7-incremental-audit.json) verifies
all five original exports, paid-label coverage, centered moments, state sizes
and retained clocks. The maximum relative ridge-equation residual was
3.03 × 10⁻¹⁵. The original experimental replay was run once; verifying the
label-bearing timing subset added an audit receipt without rerunning the
experiment.

At 192 features, centered statistics and count/position bookkeeping occupy
299,552 bytes, with 3,072 additional bytes for the fixed normalizer. The
**302,624-byte persistent calibration state** is about 152 times smaller than
the earlier runner's 46,080,000-byte full-prefix representation cache. This is
a comparison of numeric components, not application RSS or a whole-system
memory reduction. Explicit raw/target chunk buffers add 99,328 bytes. Computing
the initial normalizer temporarily used a fixed 2,304,000-byte, 30-second
feature buffer, which was then discarded. The causal filter adds 3,072 bytes.
Arithmetic scratch, Python overhead, model arrays and the validation runner's
loaded reference data are separately outside these calibration-state counts.

State grows as \(O(d^2)\), so channel count still matters. It does not grow with
label count. Changing the normalizer or forgetting old observations would
require a different state update rule; this implementation specifically serves
the fixed-normalization, accumulating-label policy.

## Update work and solve work

The engineering replay retained raw wall and process-CPU clocks for every
update and for each solve/export/reload. It performed 400 bounded updates, of
which 163 added at least one eligible label. Causal feature generation and input
loading occurred outside the update timer.

| Operation | Repetitions | Median wall time | Maximum wall time | Total process CPU time |
|---|---:|---:|---:|---:|
| Statistics update, all chunks | 400 | 0.0155 ms | 0.345 ms | 21.901 ms |
| Statistics update, chunks adding labels | 163 | 0.109 ms | 0.345 ms | 18.725 ms |
| Ridge solve | 5 | 0.178 ms | 0.204 ms | 0.917 ms |
| Solve plus export/reload | 5 | 1.005 ms | 1.312 ms | 4.664 ms |

The statistics update distributes work while labels arrive; the final solve
uses the accumulated state. These timings are one source-only engineering
replay, not a controlled speedup comparison with the earlier ten batch refits.
They show that this representative workload has ample margin within the
evaluation's two-second refit allowance. They do not establish worst-case
realtime bounds, reduce the scientific label budget, or demonstrate that
continual recalibration improves fresh accuracy.

The [accumulator](incremental.py), [source validation runner](incremental_runtime.py)
and [synthetic tests](../../tests/neural_decoder/test_v7_incremental.py) are
separate from the frozen V7A and V7B producer paths. Private sufficient-statistic
snapshots, models, row-inventory receipts and raw clocks remain local.
