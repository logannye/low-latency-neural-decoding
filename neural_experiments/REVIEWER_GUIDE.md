# Neural decoding evidence guide

Start with the original [H1 final report](FINAL_REPORT.md): a 23,367-parameter
causal TCN reached mean day R² 0.177583 versus 0.034146 for selected ridge across
seven later days from one participant. All 21 seed/day comparisons favored TCN.
These are frozen historical local FALCON results, not official benchmark scores.

The [final deployment protocol](H1_FINAL_CLOSURE_PROTOCOL.md) and its
[pinned configuration](configs/h1-final-closure-v1.json) address the remaining
link between these exact final weights and conversion, runtime, and recovery.
Seed 0 is fixed for timing and has historical R² 0.177086; the three-seed mean
must not be paired with its timing as if it were one model. The original speed
and compression studies use separately identified development weights. New
final-weight measurements must be read from their own completed report and audits.
The new study reuses exposed recordings for deployment validation.

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
