# V4: useful history reuse, unresolved T3 failure

The locked study establishes useful aggregate decoding with **zero fresh labels**
and a 30-second unlabeled normalization prefix across 23 newly accessed sessions
from eight known participants. Mean R² is **0.238 [0.105, 0.339]**, with seven of
eight participant means positive. Adding 30 seconds of fresh labels gives
**0.245 [0.118, 0.352]** and passes the predeclared useful-accuracy gate.

The stronger claim that 30 seconds retains 120-second daily accuracy **does not
pass**: the paired difference is **−0.029 [−0.083, 0.018]**, crossing the declared
−0.05 margin. T2 remains negative under history reuse. T3's separate continuation
test also **fails recovery**: the targeted repair reduces error but remains worse
than the zero-vector control on average. No configuration was changed after these
fresh outcomes were evaluated.

## New-session results

All methods score identical valid pairs from 122 observed seconds through
`min(600 seconds, duration)`. Session scores are averaged within participant,
then eight participants receive equal weight. Intervals resample participants
10,000 times with seed 20261023. They are descriptive intervals for this small,
previously studied cohort.

| Procedure | Fresh labeled prefix | Mean R² [95% interval] | Positive people | Angular error |
| --- | ---: | ---: | ---: | ---: |
| Zero output | 0 s | −0.038 [−0.063, −0.018] | 0/8 | 90.00° |
| Prefix mean | 30 s | −0.165 [−0.268, −0.083] | 0/8 | 87.77° |
| V3 daily refit | 30 s | −0.021 [−0.195, 0.139] | 4/8 | 67.75° |
| Matched scratch | 30 s | 0.069 [−0.025, 0.168] | 6/8 | 66.45° |
| History prior | 0 s | **0.238 [0.105, 0.339]** | **7/8** | **55.00°** |
| History + adaptation, primary | 30 s | **0.245 [0.118, 0.352]** | **7/8** | **53.77°** |
| V3 daily refit | 120 s | 0.275 [0.171, 0.385] | 8/8 | 51.41° |
| History + adaptation, secondary | 120 s | 0.299 [0.177, 0.403] | 7/8 | 50.27° |

![Participant results](../figures/v4-reduced-calibration.png)

The primary history arm improves on matched scratch by **+0.176 R²
[0.110, 0.229]** with the same fresh labels, representation, normalization and
adaptation penalty. This supports the contribution of historical supervised
weights. It does not imply learning without labeled data: history can include
up to three earlier 600-second labeled recordings, and the query needs 30 seconds
of unlabeled features before its normalization is available.

| Participant | History, 0 s labels | History, 30 s labels | Daily, 120 s labels |
| --- | ---: | ---: | ---: |
| T2 | −0.157 | −0.127 | 0.032 |
| T5 | 0.480 | 0.494 | 0.541 |
| T6 | 0.230 | 0.228 | 0.204 |
| T7 | 0.295 | 0.313 | 0.269 |
| T8 | 0.315 | 0.326 | 0.466 |
| T9 | 0.275 | 0.319 | 0.291 |
| T10 | 0.231 | 0.153 | 0.155 |
| T11 | 0.237 | 0.259 | 0.239 |

The average therefore does not justify enabling history reuse for everyone.
T2 has negative transfer relative to its daily120 comparator. The 30-second
fresh labels also reduce T10's mean relative to zero-label history reuse. These
are retained failures, not exclusions or triggers for post-assessment retuning.
The 120-second history result is a prespecified secondary arm, not a replacement
for the primary reduced-calibration endpoint.

## T3 continuation results

T3 has no unused session identities in this release. The test uses untouched
later blocks in eight known sessions, with each query fitted only from its
initial 30/120-second prefix. The first two whole blocks starting at or after
600 seconds were fixed by metadata; day62 has only one eligible block, retained
without substitution. The domains are never pooled into an accuracy headline.

| Procedure | Fresh labels | Mean R² [session-bootstrap interval] | Positive sessions | Angular error |
| --- | ---: | ---: | ---: | ---: |
| Zero output | 0 s | −0.046 [−0.108, −0.010] | 0/8 | 90.00° |
| V3 daily refit | 120 s | −0.351 [−0.518, −0.202] | 0/8 | 87.92° |
| Targeted repair | 120 s | **−0.087 [−0.163, −0.023]** | **0/8** | **87.53°** |
| Targeted repair | 30 s | −0.090 [−0.172, −0.025] | 0/8 | 86.94° |
| History prior | 0 s | −0.194 [−0.279, −0.105] | 0/8 | 92.64° |
| History + adaptation | 30 s | −0.230 [−0.314, −0.143] | 0/8 | 90.55° |
| History + adaptation | 120 s | −0.215 [−0.373, −0.079] | 0/8 | 89.83° |

![T3 continuation results](../figures/v4-t3-continuation.png)

The targeted120 repair gains **+0.263 R² [0.082, 0.456]** over V3 daily120,
but its difference from zero is **−0.041 [−0.104, 0.003]**. None of its session
scores is positive. This is error reduction without useful recovery. The
bootstrap unit is a session within one person, not an independent participant.

The [exposed-data diagnosis](DIAGNOSIS.md) identifies sparse high-threshold
counts, unstable power features, limited directional coverage and overfitting.
Lowering the detection threshold adds many events but did not recover useful
suffix information in the bounded pilot. Weakening normalization floors or
regularization increased fit accuracy while worsening later predictions.
The fresh continuation result is consistent with these limitations; it does
not prove that T3 contains no decodable information or identify a biological
cause of its poor signal.

## Evaluation and implementation evidence

- The acquisition and full experiment plans were committed before fresh outcome
  preparation. All 24 requested new members were obtained. T6 day1034 is only
  113.90 seconds and remains unavailable on the common window; 23 sessions score.
- All 80 old sessions contribute source pseudo-queries. Independent verification
  checked 2,560 numeric model exports/full prediction traces, nine global choices
  and eight T3 choices. The maximum source replay difference was
  **1.77 × 10⁻¹⁴**. Eighty targeted source rows lack sufficient suffix evaluation
  data and retain null scores; no candidate- or outcome-dependent removal occurs.
- Source choices and their passed audit were published at `b400c8a` before fresh
  assessment. Global choices use entire-person exclusion; T3 targeted choices
  use strictly earlier T3 days, or non-T3 data for the earliest session.
- History selection checks numeric acquisition day and electrode layout, not
  archive order. No query or other fresh assessment session trains its prior.
  Downstream replay uses no future observations, inference is label-free, and
  all fits reload numeric exports before predicting.
- The fresh ledger contains 272 requested method/session cases, including eight
  unavailable rows for the short T6 session. All 264 saved models and full traces
  passed independent assessment audit; maximum prediction discrepancy was
  **1.78 × 10⁻¹⁴**. No fresh fit needed the insufficient-label fallback. The
  first audit attempt rejected enriched archive
  identity fields; the auditor was corrected to verify those fields against the
  acquisition receipt. Its failed log and exact source snapshot remain local.
  No model, prepared array, selection, prediction or score changed.
- Local tests, hosted CI, model-fit equations, full prediction replay, bootstrap
  arithmetic and runtime timestamps provide separate verification layers.
  See [validation](../results/v4-validation.json), [source choices](../results/v4-source-selection.json),
  [cohort](../results/v4-cohort.json) and [complete results](../results/v4-study.json).

## Streaming cost

The protocol chose T2 day187, archive rank10, `selected_prior30` before consulting
its accuracy. The exact export uses TX4.5 plus power, 200 ms smoothing, no slow
baseline correction, and adaptation penalty10. All prescribed blocks are retained.

| Replay | Updates | Processing p99 | Release-to-completion p99 | 20 ms deadline misses |
| --- | ---: | ---: | ---: | ---: |
| Three fast blocks | 3 × 30,000 | 0.01975–0.02467 ms | 0.01983–0.02475 ms | 0 |
| Paced block1 | 3,000 | 0.84974 ms | 5.19847 ms | 0 |
| Paced block2 | 3,000 | 1.07774 ms | 5.20462 ms | 0 |

Scheduling matters: paced processing is slower than the tight replay loop, and
queue p99 is 4.863–4.884 ms. Maximum processing time reaches 3.845 ms. The
processing p99 remains below2ms, but this is a finite ordinary-OS benchmark,
not a worst-case or hard real-time guarantee. Acquisition, transport and upstream
feature extraction are outside the measured boundary.

The model has 386 readout parameters, 6,160 bytes of numeric model arrays and
3,072 bytes of persistent numeric filter state. Process high-water RSS is about
399.7 MiB and includes loaded replay arrays and parity workspaces; it is not
the model footprint. Batch/stream maximum error is **7.61 × 10⁻¹⁵**. Independent
integer-timestamp verification passed all five blocks. The 200ms fast filter has
190.17ms steady-state mean sample age and a 30-bin/600ms 95% step response;
those descriptors are separate from computation and scheduling latency.

The [aggregate runtime receipt](../results/v4-runtime.json) binds exact model,
feature, plan and source identities. Calibrated-model whole-trace replay is a
systems check; scientific scores remain restricted to the declared post-prefix
masks. The runtime auditor reconstructs timestamps and resource arithmetic;
independent scientific replay and representative selection are separate checks.

## Implications

History reuse is a credible way to reduce fresh labeling for several participants,
but these data do not establish universal calibration-free operation or preserved
120-second accuracy. A next prospective study should test a source-calibrated
rule for accepting history reuse versus collecting additional, directionally
diverse calibration. The current T2 and T3 outcomes are now exposed and cannot be
reused as fresh confirmation for that rule.

For T3, another larger temporal model is poorly motivated by the observed
overfitting. Better evidence would come from independently acquired recordings,
explicit signal-quality and task-alignment checks, and calibration that covers
movement directions rather than merely extending correlated observations. The
current release does not support a useful T3 decoder under this locked endpoint.

These are offline predictions of cursor-to-target direction in existing
closed-loop recordings. They do not establish online control, clinical benefit,
untouched-participant generalization, or causal upstream feature extraction.
See the [protocol](PROTOCOL.md) and [reproduction guide](README.md).
