# V9 results: trial-balanced calibration and target validation

**Neither the trial-balanced readout nor longer continuous calibration rescued T2.** Increasing continuous calibration from 120 to 360 recorded seconds improved every fresh T2 session, but only one became positive and mean error remained approximately equal to zero output. The trial policy achieved its coverage quota in five of six T2 sessions while its readout failed on all six. Both predeclared recovery gates failed.

This report covers the fixed, once-scored fresh assessment on 2026-10-23. The [protocol](PROTOCOL.md), [study plan](study-plan.json), [source selection](source-selection.json), [data contract](DATA.md), and [reproduction guide](README.md) describe the frozen design. [Public aggregate results](../results/v9-study.json) retain all methods and cases.

## What was tested

The hypothesis was that insufficient trial diversity and an imprecise target definition explained T2's earlier failure. The policy recruited at least 24 completed eligible trials with at least four in each cardinal-centered direction sector, stopping by 390 recorded seconds. Labels start at second 30, so the maximum purchased interval is 360 seconds. Trials must have complete, stable, coherent target geometry in a common window that supports both reaction-offset candidates.

The provider-informed training target is the unit vector of mean cursor-to-target displacement over a 300 ms window after a reaction offset. It is not the direction at onset plus 40 ms; that reference controls far-distance eligibility. The provider source and retained data geometry were checked independently. Actual intended movement, task identity, assistance and success are not established by these fields. This is not an exact reproduction of the provider's dSNR pipeline.

The primary trial readout uses one neural feature vector per completed trial, derived from a causal 300 ms boxcar. The source screen chose reaction offset 0.2 seconds and ridge 1 from four fixed options. Continuous controls retain the original 200 ms exponential filter and ridge 1. Twelve methods distinguish the quota stop, extra calibration duration, weighting, target and negative controls. Feature-only prediction receives no cursor, target or trial phase.

All methods share the continuous endpoint and primary suffix [392,600) seconds. Every go interval overlapping the maximum purchased interval [30,390) is excluded from all methods. Valid go bins must pass block burn-in. Secondary trial-window scores use a common eligible inventory and cannot satisfy the primary recovery criterion.

## Cohort, exposure and provenance

The fixed fresh cohort contains **24 sessions, six each from T2, T7, T9 and T10**, at regular archive ranks 25–30. All 24 were acquired, prepared and scored. There were no failed archive attempts, shortages, substitutions, dropped sessions or neural-readout fallbacks. One recording, T10 day 216, contains 498.34 seconds; the others provide the full 600-second prefix. All 714,917 prepared paired bins passed the separate raw-context reconstruction audit.

Archive order is not chronological day order. Fresh T2 days are **46, 382, 411, 418, 424 and 511**. Day 46 is included under the original selection rule; it was not removed after its unfavorable outcome. These are previously unopened recordings from known people, not a strictly later-day cohort or untouched-person validation.

Acquisition transferred **12,968,525,824 compressed bytes** and retained **2,647,237,885 raw bytes**, within the 32 GB/12 GB limits. Earlier members were authenticated against both pinned V8 parent indexes. Partial archive streaming authenticates member hashes but does not verify the complete archive digest. [Cohort receipt](../results/v9-cohort.json).

The acquisition lock was committed at `29b96b9`; scientific code and the plan were frozen at `2d0edbf` before source fits. Development reused 200 exposed prefixes and fitted models on 96 of them. Nine exposed T2 sessions selected the recipe. The source audit verified every one of 1,188 models, then the sealed selection was committed and pushed at `0d6b80a` before fresh conversion. No scientific source changed between source fitting and fresh assessment. This is an auditable committed predeclaration, not a claim of external preregistration.

## Primary held-out accuracy

Each cell is an equal-session mean R². Cohort summaries first average within each person, then give people equal weight. Negative R² means more squared error than the held-out mean-direction reference; it is not a probability or percentage of correct commands.

| Fixed method | T2 | T7 | T9 | T10 |
|---|---:|---:|---:|---:|
| baseline | -0.1709 | 0.3223 | 0.1138 | 0.3349 |
| duration | -0.1076 | 0.4060 | 0.4052 | 0.4023 |
| quota_continuous | -0.1573 | 0.3108 | 0.2054 | 0.3474 |
| trial_startup_sector_mean | -0.3799 | 0.0940 | -0.3424 | 0.1315 |
| trial_duration_sector_mean | -0.4128 | 0.1408 | 0.0078 | 0.1900 |
| quota_uniform_endpoint | -0.5000 | 0.1134 | -0.2125 | 0.1243 |
| quota_sector_endpoint | -0.4984 | 0.1139 | -0.2228 | 0.1189 |
| quota_uniform_mean | -0.4985 | 0.1111 | -0.2133 | 0.1285 |
| quota_sector_mean | -0.4969 | 0.1115 | -0.2237 | 0.1230 |
| zero | -0.1007 | -0.0425 | -0.1325 | -0.0978 |
| constant | -0.1118 | -0.0195 | -0.0651 | -0.0491 |
| trialshuffle | -0.3644 | -0.4189 | -0.6488 | -0.6064 |

The continuous 360-second control reaches mean R² **0.2765** across all four participants (participant-bootstrap interval **[0.0206, 0.4056]**) and **0.4045** across the three comparison participants. Its three-person mean rises from **0.2570** under startup calibration. These are matched comparisons on the same fresh cases and scoring masks; earlier-version headline scores used different cohorts and are not matched comparators.

T2's longer continuous calibration has mean R² **−0.1076**, versus **−0.1709** at startup and **−0.1007** for zero output. The paired gain over startup is **+0.0633 [0.0314, 0.1030]**, with improvement on all six sessions. Nevertheless, only one session is positive, only two beat zero, and mean summed MSE is **1.0010**, versus **1.0000** for zero output. Consistent improvement relative to a weak starting decoder is not useful recovery.

## Every fresh T2 session

| Post-implant day | Startup | Duration | Quota continuous | Trial primary | Zero | Quota labels (s) | Quota |
|---|---:|---:|---:|---:|---:|---:|---|
| 46 | -0.5856 | -0.5610 | -0.6710 | -1.3583 | -0.4445 | 214.96 | met |
| 382 | -0.1256 | -0.1021 | -0.2840 | -0.3440 | -0.0595 | 228.98 | met |
| 411 | -0.0436 | 0.1105 | 0.1082 | -0.3673 | -0.0226 | 339.44 | met |
| 418 | -0.0658 | -0.0202 | -0.0224 | -0.2739 | -0.0072 | 355.06 | met |
| 424 | -0.1314 | -0.0449 | -0.0468 | -0.2506 | -0.0293 | 321.46 | met |
| 511 | -0.0731 | -0.0278 | -0.0278 | -0.3873 | -0.0409 | 360.00 | timeout |

Both primary gates fail: useful recovery requires all six scorable, mean R² at least 0.05, at least four positive sessions and paired gains with strictly positive interval lower bounds over startup and zero. The incremental-benefit gate additionally requires a similarly positive gain over the full-duration decoder. The fixed trial primary has **0/6 positive sessions**; no fresh comparison replaces it after inspection.

## Calibration cost and what the quota establishes

| Participant | Mean purchased labels (s) | Median (s) | Quota attained | Positive trial-primary sessions |
|---|---:|---:|---:|---:|
| T2 | 303.32 | 330.45 | 5/6 | 0/6 |
| T7 | 106.79 | 108.25 | 6/6 | 4/6 |
| T9 | 103.18 | 87.67 | 6/6 | 4/6 |
| T10 | 109.11 | 108.97 | 6/6 | 4/6 |

The quota is reached in 23/24 sessions overall. T2 day 511 times out with 23 eligible trials and sector counts [5,5,7,6]: it lacks one total trial, not a direction sector. It remains in every result. The quota-continuous control purchases exactly the same intervals as the trial primary.

T2 uses 303.32 seconds of labels on average: 15.75% less than the full 360-second budget but 2.53 times the 120-second startup budget. These are label costs. Time until the declared deployment point also includes 30 seconds of normalization and a two-second allowance. That allowance is not a measured V9 refit latency. The common late evaluation omits the startup model's earlier availability, so this experiment does not establish a whole-session control or utility advantage.

Five T2 sessions satisfy both the count and coverage requirements, yet none has positive trial-primary R². **Quota attainment establishes coverage under a proxy definition, not decoder acceptance.** The study's outcome gates reject recovery; a deployment policy must not treat the collection rule as an accuracy guarantee.

## Matched mechanisms and remaining uncertainty

All contrasts below use the same cases and primary masks. Intervals resample the six T2 sessions with 10,000 bootstrap replicates and the fixed seed. They are descriptive session intervals within one person, not participant replication or multiplicity-adjusted mechanism tests. Possible dependence between sessions is not modeled by this bootstrap.

| T2 paired comparison | Mean R² difference | Session-bootstrap 95% interval |
|---|---:|---|
| primary_minus_baseline | -0.3260 | [-0.5257, -0.1863] |
| primary_minus_zero | -0.3962 | [-0.6163, -0.2630] |
| primary_minus_duration | -0.3893 | [-0.5725, -0.2514] |
| duration_minus_baseline | +0.0633 | [+0.0314, +0.1030] |
| primary_minus_quota_continuous | -0.3396 | [-0.5074, -0.1876] |
| sector_weighting | +0.0016 | [-0.0011, +0.0055] |
| window_target | +0.0015 | [-0.0005, +0.0048] |
| full_budget_minus_quota | +0.0840 | [-0.0073, +0.2222] |
| quota_minus_startup_trial | -0.1170 | [-0.1937, -0.0275] |

The primary loses **0.3396 R²** relative to continuous training at exactly the same quota stop. This comparison bundles temporal pooling, phase sampling and label representation; it does not isolate which component causes the loss. More purchased time helps the continuous decoder, but adds recency and potentially new trials as well as label quantity.

Equal-sector weighting and the averaged target each change T2 by less than 0.002, with intervals spanning zero. The coherence/stationarity filter deliberately chooses trials where the endpoint and averaged direction are similar. Their small difference is evidence that this specific revision is insufficient, not proof that the original target equals intention.

The primary's matched held-out trial-window R² remains **−0.2453** for T2. Thus failure is not solely explained by evaluating outside the training phase. The same secondary means are **0.3176, 0.3199 and 0.3701** for T7, T9 and T10; these changed-endpoint diagnostics cannot replace continuous performance. The single cyclic trial-label shuffle is a negative control, not a permutation distribution or significance test.

The exposed source evidence predicted difficulty: the chosen recipe averaged −0.2655 on the nine selection sessions. Across all 24 exposed T2 sessions, more continuous calibration improved R² from 0.0363 to 0.1204, but the latest nine still averaged −0.0547 under that control. These source numbers are development results, never added to the six-session fresh evidence. [All source results](../results/v9-development-study.json).

A justified next hypothesis is neural/behavioral timing or task-state mismatch. A future study could retain continuous labels and the full-duration control, select a small causal temporal adjustment using whole-trial splits inside calibration, and freeze it before another untouched cohort. No such experiment was run here. If calibration-held-out signal remains weak, richer task, assistance and intention metadata is more informative than declaring a neural limitation or expanding the network blindly.

## Verification and evidence boundaries

The independent source audit passed **1188 models / 105 trace bundles**, and the fresh audit passed **288 models / 24 trace bundles** on their first real attempts. The fresh maximum prediction difference is **2.9e-13** and normal-equation residual **3.04e-13**. Reconstruction covers training masks, weights, targets, normalization, quota stopping, model exports, full predictions, both scoring endpoints, source choices, bootstrap summaries and recovery gates. [Source audit](../results/v9-source-audit.json), [fresh audit](../results/v9-assessment-audit.json), [raw-data audit](../results/v9-fresh-data-audit.json).

Synthetic tests cover future-label and feature perturbations, complete-trial boundaries, the raw 389.99-second onset edge, block resets, streaming/export parity, altered artifacts, exact selection binding and publication privacy. The final local Python suite passed 1,321 tests with 22 environment-dependent skips; Ruff lint and formatting passed. Hosted CI validates the published revision separately. Public artifacts contain aggregate evidence and hashes; raw samples, trial IDs, predictions, model coefficients and signed transport links remain local and ignored.

This is offline evidence. It establishes neither closed-loop performance nor clinical benefit, zero-shot person transfer, actual intention, upstream feature-extraction causality, acquisition latency or a hard real-time deadline. V8 timing measurements remain explicitly versioned historical evidence; V9 introduces no new runtime benchmark claim. T2 and T3 remain unresolved.
