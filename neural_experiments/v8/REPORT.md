# V8: T2 failure diagnosis on fresh sessions

**Predeclared T2 recovery gate: FAIL.** The source-selected primary was `baseline`. The source screen retained the existing baseline among the neural recipes; no fresh result changed that choice.

The fixed cohort contains 24 selected sessions, with 23 scorable sessions across 8 known participants. T2's primary mean R² is -0.008. Every selected case, including unavailable scores and failed interventions, is retained below and in the public artifacts.

## What was tested

The factorial experiment separates two neural representations, all-phase versus early/far training labels, and uniform versus equal-trial weights. The original continuous cursor-to-target direction remains the primary endpoint. A dedicated stable-reference target head and its matched instantaneous-target head are also evaluated on exactly the same early/far bins. Changing that secondary endpoint cannot satisfy the primary gate.

Every method purchases the same recorded-time interval, seconds 30–150, after 30 seconds of normalization. Scoring starts at 152 seconds, ends by 600 seconds, and excludes all trials overlapping calibration. Each session trains its own readout. The recipes were selected on already exposed T2 data, so this is known-participant session generalization, not zero-shot transfer to a new person.

## All fixed methods

| Method | T2 mean R² | Cohort mean R² [participant 95% interval] | Other seven mean R² |
|---|---:|---:|---:|
| Baseline | -0.008 | 0.303 [0.194, 0.398] | 0.348 |
| Baseline + equal-trial weights | -0.006 | 0.300 [0.191, 0.396] | 0.344 |
| Baseline + early/far training | -0.213 | 0.169 [0.000, 0.317] | 0.224 |
| Baseline + early/far + equal-trial | -0.213 | 0.170 [0.001, 0.319] | 0.225 |
| TX3/500 ms representation | -0.045 | 0.243 [0.132, 0.340] | 0.284 |
| TX3/500 ms + equal-trial | -0.034 | 0.243 [0.133, 0.342] | 0.283 |
| TX3/500 ms + early/far | -0.158 | 0.131 [-0.009, 0.260] | 0.172 |
| TX3/500 ms + early/far + equal-trial | -0.158 | 0.132 [-0.009, 0.261] | 0.174 |
| Zero output | -0.085 | -0.051 [-0.089, -0.019] | -0.046 |
| Purchased-label constant | -0.028 | -0.017 [-0.021, -0.014] | -0.016 |
| Fixed trial-label rotation | -0.230 | -0.363 [-0.450, -0.280] | -0.381 |
| Stable-reference target head | -0.219 | 0.157 [-0.014, 0.305] | 0.211 |

Average sessions within each person, then average people equally. Confidence intervals resample people; the three-session T2 intervals are descriptive and provide limited uncertainty information. Other than the locked primary, these are prespecified diagnostic comparisons, not alternate winners selected after testing.

![Fresh T2 mechanism comparisons and every participant](../figures/v8-diagnosis.png)

## Every T2 session

| Session | Baseline | TX3/500 ms | Equal-trial | Early/far | Stable target head | Zero |
|---|---:|---:|---:|---:|---:|---:|
| T2_day_333_decoding | -0.004 | -0.023 | 0.018 | -0.275 | -0.267 | -0.133 |
| T2_day_347_decoding | -0.098 | -0.161 | -0.109 | -0.171 | -0.189 | -0.090 |
| T2_day_361_decoding | 0.079 | 0.051 | 0.073 | -0.191 | -0.202 | -0.031 |

## Factorial attribution

| Paired effect | T2 mean change in R² | Cohort mean change [participant 95% interval] |
|---|---:|---:|
| representation | 0.0110 | -0.0484 [-0.0827, -0.0170] |
| clean phase | -0.1622 | -0.1218 [-0.2022, -0.0571] |
| equal trial | 0.0031 | -0.0002 [-0.0023, 0.0016] |
| representation x phase | 0.0871 | 0.0204 [-0.0030, 0.0473] |
| representation x weighting | 0.0041 | 0.0016 [-0.0005, 0.0035] |
| phase x weighting | -0.0062 | 0.0030 [-0.0020, 0.0081] |
| three way | -0.0081 | -0.0034 [-0.0068, -0.0001] |

Main effects average the paired change across the other two factors. Interactions are differences of differences. These are descriptive mechanism comparisons without multiplicity-adjusted significance claims. Equal-trial weighting changes duration dominance; it does not collect new trials or balance directions. Early/far training reduces the amount of usable data within the same purchased interval, so its effect combines phase choice and information quantity.

## Known-participant portability

| Person | Selected/scorable | Baseline R² | TX3/500 ms R² |
|---|---:|---:|---:|
| T2 | 3/3 | -0.008 | -0.045 |
| T5 | 3/3 | 0.524 | 0.432 |
| T6 | 3/2 | 0.257 | 0.111 |
| T7 | 3/3 | 0.320 | 0.295 |
| T8 | 3/3 | 0.307 | 0.238 |
| T9 | 3/3 | 0.226 | 0.156 |
| T10 | 3/3 | 0.430 | 0.376 |
| T11 | 3/3 | 0.371 | 0.382 |

## Evidence and interpretation

The [mechanism diagnosis](DIAGNOSIS.md) documents the source decline, fresh confirmation, trial coverage, target geometry, feature sparsity and error decomposition. Its post-development scalar-gain oracle is an explicitly nondeployable diagnostic. It neither changes the selected model nor bounds information in the original recordings.

The data do not identify actual intention, assistance level, task strategy, electrode biology or a unique causal explanation for failure. Failure of these finite fixes cannot prove that T2 is undecodable. Conversely, more threshold crossings, positive correlation, or success on an easier subset cannot establish useful continuous control.

See the high-level repository README for the fresh mechanism interpretation.

## Reproduction and provenance

- [Frozen protocol](PROTOCOL.md) and [source selection](source-selection.json).
- [Data contract](DATA.md), [fresh cohort](../results/v8-cohort.json), and [fresh raw-context audit](../results/v8-fresh-data-audit.json).
- [All fresh aggregate results](../results/v8-study.json) and [independent numeric audit](../results/v8-audit.json).
- [Source results](../results/v8-development-study.json), [source audit](../results/v8-development-audit.json), and [runtime evidence](RUNTIME.md).
- [Reproduction commands](README.md). Raw samples, private traces and fitted coefficients remain outside Git.

Public assessment SHA256: `60b1877b4196ccd972070a4a31f30fa5711e9437d474679bf86d3368b13b971b`.
Private assessment manifest SHA256: `04865c454f5e5d04f5bded8e279776b8dc05aefb7710b3bf53a9a154ce5b145f`.
Independent audit SHA256: `6796ff40e90b0974efb104f8d949e5009ad11e2e936b44695022b7789f8d2284`.
