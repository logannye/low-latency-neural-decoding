# Neural decoding evidence guide

Start with the original [H1 final report](FINAL_REPORT.md): a 23,367-parameter
causal TCN reached mean day R² 0.177583 versus 0.034146 for selected ridge across
seven later days from one participant. All 21 seed/day comparisons favored TCN.
These are frozen historical local FALCON results, not official benchmark scores.

The [final H1 deployment report](H1_FINAL_CLOSURE_REPORT.md) connects these exact
final weights to conversion, five-path runtime and recovery evidence. All three
seeds passed the fixed int8 quality budget: the graph shrank 61.47%, with worst
seed/day variance-weighted R² change −0.001715. The float32 layout control explains
most of the fast-replay improvement. The report retains the failed overall 2 ms
service-p99 goal, isolated outliers and output-specific losses. Both 90,000-bin
soaks passed prediction/timestamp audits, retaining one NumPy and five int8
deadline misses. Inspect the [aggregate tables](results/h1-final-closure-v1/tables.md)
and [audit index](results/README.md) for every result and source binding.

Its [protocol](H1_FINAL_CLOSURE_PROTOCOL.md),
[pinned configuration](configs/h1-final-closure-v1.json), and
[reproduction commands](H1_FINAL_CLOSURE_REPORT.md#reproduction-commands) separate
conversion quality from timing. The predetermined seed-0 int8 model has later-day
R² 0.177615; the original three-seed mean must not be paired with seed-0 timing as
if it were one model. Timing uses development recordings; quality uses separately
identified later-day recordings. This validates deployment on exposed data and
does not add fresh generalization evidence. Original runtime/compression reports
retain their separately identified developmental weights.

The separate [V3 result](v3/REPORT.md) improved matched 120-second daily-calibration
R² from 0.0208 to 0.2578 over 26 sessions from nine known participants. The
[V4 result](v4/REPORT.md) combined history and 30 seconds of fresh labels to reach
0.245 versus 0.069 for matched scratch, but did not establish noninferiority to
120-second daily calibration. These tasks, models and cohorts differ from H1;
their accuracy cannot be combined with H1 timing.

[V5](v5/REPORT.md), [V6](v6/REPORT.md), [V7](v7/REPORT.md), [V8](v8/REPORT.md) and
[V9](v9/REPORT.md) preserve the failed calibration and adaptation hypotheses.
In V9, trial balancing did not recover T2, and six minutes of continuous labels
improved its mean only from −0.171 to −0.108. Neither history reuse nor quota
completion is a sufficient deployment acceptance test.

For reproduction, use the [original H1 guide](README.md), [scoring contract](SCORING.md),
[historical quality protocol](PROTOCOL.md), [evidence index](results/README.md), and
[V9 guide](v9/README.md). Public artifacts contain aggregates and audit bindings;
recordings, fitted weights, sample-level predictions and timings remain local.
All results concern offline replay. Untouched-person transfer, closed-loop benefit,
physiological response latency and implant power savings remain unestablished.
