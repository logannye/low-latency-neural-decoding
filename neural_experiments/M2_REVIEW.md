# M2 transfer review

The fixed ridge recipe failed this sparse forward-day quality check: mean day
R2 was **−13.497285**, versus **−0.034258** for the training-target mean. Ridge
was worse on every future day. Independent reconstruction passed, so this is
a retained negative modeling result, not a replay or scoring discrepancy.
No settings changed and no additional recordings or models were tried.

The successful transfer is limited to the data, fitting and causal replay
implementation. It does not establish effective cross-task decoding quality,
generalization of H1 weights, cross-participant generalization, or an official
FALCON result. The fitted M2 weights were estimated from M2 training data.

## Scope and provenance

The protocol was committed as `d80e4d0d1b8d505121eeba63f62f177176abe999`
before any M2 fitting or evaluation. The run completed on its first attempt at
`results/neural-m2/minival-transfer-v1`; raw/prepared arrays, fitted models and
all ten full-file prediction traces remain local. The public
[aggregate JSON](results/m2-transfer.json) contains scores, counts, checks and
identities. The [plan](SECOND_TASK_PLAN.md) and
[locked configuration](configs/m2-minival-transfer-v1.json) retain the exact
seven provider assets, URLs, size, SHA-256 and complete metadata.

All seven minival recordings concern the same macaque, MonkeyN, with 96
threshold-crossing inputs and two outputs: index-finger movement and the
grouped middle/ring/small fingers. Four trials from October 19 provided only
**302 bins / 6.04 seconds** for fitting. Ten trials across three future days
provided **709 bins / 14.18 seconds** for scoring. Every acquired bin in this
subset had a true evaluation mask.
[Provider dataset](https://api.dandiarchive.org/api/dandisets/000953/versions/draft/info/).

| Role | File | Trials | Scored bins | Nominal seconds |
|---|---|---:|---:|---:|
| Train | 2020-10-19 Run1 | 2 | 173 | 3.46 |
| Train | 2020-10-19 Run2 | 2 | 129 | 2.58 |
| Evaluate | 2020-10-20 Run1 | 2 | 117 | 2.34 |
| Evaluate | 2020-10-20 Run2 | 2 | 116 | 2.32 |
| Evaluate | 2020-10-27 Run1 | 2 | 141 | 2.82 |
| Evaluate | 2020-10-27 Run2 | 2 | 141 | 2.82 |
| Evaluate | 2020-10-28 Run1 | 2 | 194 | 3.88 |

The fixed recipe used history 1, regularization 0.1, 12 exponential FIR taps
with tau 12 bins, and frozen normalization fitted to the training neural bins.
Regression and the constant predictor used training targets only. Minival
overlaps provider calibration data, but no calibration assets were used, and
no date or file appears in both fitting and evaluation roles.

## Results

Day R2 concatenates scored rows from all that day's files, with variance
weighting across the two outputs. The reported mean weights the three days
equally. Negative scores are retained.

| Evaluation day | Constant R2 | Ridge R2 | Scored bins |
|---|---:|---:|---:|
| 2020-10-20 | −0.027442 | −25.042713 | 233 |
| 2020-10-27 | −0.010114 | −5.417753 | 282 |
| 2020-10-28 | −0.065219 | −10.031391 | 194 |
| Equal-day mean | −0.034258 | −13.497285 | 709 |
| Population SD across days | 0.023007 | 8.378309 | |

| Day | Constant index R2 | Constant mrs R2 | Ridge index R2 | Ridge mrs R2 |
|---|---:|---:|---:|---:|
| 2020-10-20 | −0.000592 | −0.039781 | −72.995395 | −3.004578 |
| 2020-10-27 | −0.001750 | −0.019401 | −1.500399 | −9.767362 |
| 2020-10-28 | −0.139104 | −0.047704 | −26.252888 | −6.186015 |

The short training exposure and very small future samples limit interpretation.
They do not prove the cause of the quality failure. This result cannot estimate
how the same recipe would perform with full calibration, nor justify changing
regularization or selecting a different M2 subset after seeing these scores.

## Independent checks and timing boundary

[The independent audit](audit_m2.py) imports no production decoder, feature,
loader or scoring code. It verified every source/config/model/prediction/input
hash, independently read NWB datasets with HDF5, reconstructed counts using
histograms, and matched all prepared arrays. It reconstructed training means
and scales with full-file NumPy convolution and fitted sklearn Ridge with
`alpha = 0.1 × training_rows` solely to verify the saved solution. Coefficients
and intercepts agreed within 1e-12. The maximum prediction difference across
both models and all ten traces was **2.07473e-15**. Independent sklearn
recomputed all per-day and per-output scores.

The runner additionally matched the pinned official loader exactly and its
regression helper numerically. Its direct convolution versus incremental
prediction maximum was 1.38778e-16. Five synthetic tests passed after the
committed lazy-import fix, covering endpoints, overflow/cadence rejection,
future perturbation, chunk partition, file reset, rejected target/mask
arguments, unequal-length calendar-day assembly and a synthetic NWB fixture.
The only pre-run validation failure was PyNWB attempting to create its schema
cache in the home directory; lazy imports and a local cache resolved it.

**Counts are available at target timestamp +20 ms.** This follows the pinned
M2 loader's start-bin convention, although the converter describes target
timestamps as bin right edges. Predictor causality is relative to those
released counts. It must not be described as zero acquisition delay relative
to the target timestamp. The stored targets are provider `AU` values from
a per-sample numerical gradient, not physical units per second.
[Loader](https://github.com/snel-repo/falcon-challenge/blob/770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f/falcon_challenge/dataloaders.py),
[converter](https://github.com/snel-repo/falcon-challenge/blob/770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f/preproc/m2_preproc.py).

No M2 performance campaign ran and no M2 speed claim is made. The transfer
added 124,914,484 bytes including raw files, run artifacts and audit, below the
402,653,184-byte cap. Its local manifest SHA-256 is
`39369cd07d3273805ed85d58c9ae21e75fc7ded45eb5f6984ca794fa97553507`;
the independent audit receipt SHA-256 is
`4e698ae0af2173f8faf27bc45394336a2279fe12f7da9123519154b78e396c85`.
