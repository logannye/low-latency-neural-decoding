# V8 development diagnosis

This is a descriptive interpretation of the frozen development artifacts, completed before fresh assessment. Its source choices and model outcomes are preserved in the [public development study](../results/v8-development-study.json). No fresh arrays or retained tails informed this analysis. Means below give equal weight to sessions. The earlier 15 T2 sessions selected the representation; the later six selected the training policy. Both sets are exposed development data, not independent confirmation of the selected method.

## What the existing comparisons establish

T2 retains decodable signal in its earlier records, but the later problem is not solved by the tested target, weighting or feature changes. Baseline mean primary R² is **0.1716** over the earlier 15 sessions and **−0.0400** over the later six. A post-hoc chronological description localizes the decline more sharply: the first nine records, days103–172, average **0.2988**, while the next six, days187–236, already average **−0.0191**. This split describes the existing results; it is not an additional selection or hypothesis test.

| Method | Earlier15 primary R² | Later6 primary R² | Later6 angular error | Later6 centered prediction–target correlation |
|---|---:|---:|---:|---:|
| TX4.5+SBP,200ms baseline | 0.1716 | −0.0400 | 76.96° | 0.1555 |
| Selected TX3+SBP,500ms | 0.1774 | −0.0426 | 75.54° | 0.1874 |
| Baseline,equal-trial weighting | not run in this stage | −0.0573 | 77.76° | 0.1528 |
| Baseline,early/far training | not run in this stage | −0.1641 | 86.55° | 0.0220 |
| Selected features,early/far training | not run in this stage | −0.1236 | 87.29° | 0.0163 |
| Baseline,early/far stable-onset target | not run in this stage | −0.1637 | 86.65° | 0.0225 |
| Zero control | not run in this stage | −0.0232 | 90.00° | undefined |
| Purchased-label constant control | not run in this stage | −0.0344 | 83.46° | approximately0 |
| One fixed whole-trial label rotation | not run in this stage | −0.2451 | 93.44° | −0.0749 |

The selected representation changes both threshold family and filter timescale. Its small earlier advantage does not identify a benefit from lowering threshold alone: TX3+SBP at the same200ms baseline timescale averages0.1549, below TX4.5's0.1716. The later-six primary selection correctly retains the original baseline. The fixed label rotation is one negative control, not a permutation null distribution or significance test.

## Target definition

The instantaneous and stable reference targets are geometric proxies, not independently measured intention. In later T2, a mean19.43% of scored primary rows are currently near the provider's distance threshold. Across sessions, the mean of each session's median instantaneous-to-reference angle is12.05°; its90th-percentile angle averages61.00°. Geometry can matter for individual rows, but the early/far stratum makes the targets very similar: mean session-median difference1.11°, with no clean-test row near the cutoff in the latest example,day313.

Even on the identical clean secondary mask, changing the matched clean baseline from instantaneous-label training to stable-onset-label training changes mean onset R² only from **−0.1854 to −0.1821**. Corresponding trial-averaged onset R² changes from−0.0977 to−0.0942. The small change is not a recovery. The all-phase baseline's clean-stratum instantaneous/onset R² values are−0.1225/−0.1258. Thus near-target target rotation alone does not explain the tested failure. It remains possible that neither proxy captures the intended command or that the informative neural phase differs; these files cannot establish task type, assistance, reaction time or actual intention.

Restricting training is also a severe data reduction. Later-six clean fits use only320–400 rows,8–10 trials and6.4–8.0seconds of selected observations within the same120seconds purchased. All-phase fits use4,641–5,998rows. Clean weighting and uniform weighting are identical here because the retained complete clean trials each contribute40bins. The failure of clean training therefore does not isolate a target-noise mechanism from sample scarcity or an inappropriate phase window.

## Coverage and recording representation

| T2 diagnostic,equal-session mean | Earlier15 | Later6 |
|---|---:|---:|
| Purchased valid-go trials | 11.07 | 10.33 |
| Effective equal-duration trials | 9.80 | 9.05 |
| Normalized eight-sector entropy | 0.8536 | 0.8989 |
| Train/test sector total variation | 0.2935 | 0.2320 |
| Currently near-target test fraction | 29.96% | 19.43% |
| TX4.5 nonzero channel-bin fraction | 11.00% | 2.23% |
| TX3 nonzero channel-bin fraction | 47.68% | 35.12% |

Coverage remains limited by the small number of independent go intervals, but it improves rather than worsens between these two groups while decoding deteriorates. Reweighting existing trials does not create missing independent trials or directions. The much lower TX4.5 activity is consistent with a recording/threshold representation problem, but density is not a measure of information: lowering the threshold restores many counts without restoring useful held-out performance. These data do not identify electrode failure, physiological loss, threshold-setting drift or task strategy as the cause.

The later-six individual results prevent the group mean from hiding counterexamples:

| T2day | Train trials | Direction entropy | TX4.5 nonzero | Baseline R² | Selected TX3/500ms R² | Zero R² |
|---|---:|---:|---:|---:|---:|---:|
|237|10|0.8559|2.49%|0.0332|0.0351|−0.0379|
|242|10|0.9252|3.46%|−0.0226|−0.0296|−0.0228|
|250|11|0.8926|1.56%|0.0236|0.0629|−0.0106|
|257|11|0.9404|0.53%|−0.0755|−0.0749|−0.0053|
|298|9|0.8152|4.14%|−0.1379|−0.1088|−0.0035|
|313|11|0.9642|1.21%|−0.0608|−0.1404|−0.0591|

The diagnostics include other people but the supplied development model manifest contains T2 fits only. The following comparison matches full600-second recording duration, not task or participant characteristics; its values are descriptive medians across sessions. Participants have distinct coordinate units and provider distance thresholds.

| Person | Full600s sessions | Purchased go trials | Direction entropy | Train/test sector TV | TX4.5 nonzero | Near-target test fraction |
|---|---:|---:|---:|---:|---:|---:|
|T2late6|6|10.5|0.9089|0.2424|2.02%|18.93%|
|T3|8|9.5|0.7927|0.3037|0.36%|14.23%|
|T5|13|51|0.9557|0.1205|10.90%|50.52%|
|T6|5|45|0.9626|0.0852|0.72%|65.94%|
|T7|21|33|0.9512|0.1870|6.14%|42.77%|
|T8|21|41|0.9489|0.1666|7.74%|34.11%|
|T9|19|23|0.9328|0.1996|4.17%|37.75%|
|T10|21|37|0.9511|0.1263|17.57%|51.21%|
|T11|19|33|0.9590|0.1272|9.35%|47.89%|

T2's120-second budget buys far fewer independent trials than most comparison participants. Yet earlier T2 succeeds with a similarly small count, and T2's near-target fraction is lower than most others. Neither observation alone establishes the cause. No comparative decoding-performance claim about those other people follows from this diagnostic-only table.

## Output scale: the most actionable narrow adjunct, with a limited promise

For the later-six baseline, mean squared target magnitude is1.0000, mean squared prediction magnitude is0.1647, and twice the target–prediction dot product is0.1475. The identity MSE=target energy+prediction energy−2×alignment therefore gives **1.0172**, worse than the existing zero control's1.0000. The selected TX3/500ms decoder improves mean centered correlation from0.1555 to0.1874 and angular error by1.42°, but increases prediction energy to0.2121 against2×alignment0.1930; its MSE remains1.0191. Mean squared bias is only0.0317 for the baseline, while centered residual error is0.9854. A universal offset correction cannot account for most of the loss; day298 has a more prominent mean-shift contribution(0.0890).

This is amplitude too large relative to weak alignment, not a claim about calibrated probabilistic confidence. Shrinkage or abstention is an actionable, low-complexity adjunct to test with training-only, trial-disjoint validation. It might reduce harmful output energy and beat the zero/constant controls, but neither its benefit nor a deployable tuning rule has been demonstrated here. A subsequent post-development diagnostic computed the best scalar in [0,1] separately for each session using its held-out target dot products. Even this optimistic, nondeployable ceiling reaches only mean R²0.0167 for the baseline and0.0278 for the selected representation, below the0.05 usefulness criterion. This upper bound applies only to scaling these existing predictions; it is not a bound on neural information. The [oracle artifact](../results/v8-development-gain-oracle.json) preserves every case. Scaling cannot repair angular errors or produce neural information. The modest positive centered correlation in every later baseline session(0.0616–0.2495), better angular error than zero, and worse fixed-label-rotation control argue against claiming complete absence of signal; the negative aggregate R² and modest directional accuracy argue equally strongly against claiming useful decoding has been recovered.

A careful next conclusion is: **the tested target cleaning, trial reweighting and denser neural features did not rescue T2; the later records retain weak predictive alignment but output variance overwhelms that alignment. A conservative output-amplitude policy is worth a separately locked test as harm reduction, while useful recovery may require richer independent trial coverage, an appropriate target/phase contract or additional neural information.** The frozen fresh primary assessment is preserved in the [V8 results report](REPORT.md); no oracle scaling enters its scored predictions or recovery gate.

## Exact inputs

- `development-diagnostics-final.json`: SHA256 `8e441e3b20f4801556819c4b4d6da558503a12a27645e3ea24708c7a949e3190`.
- `development/manifest.json`: SHA256 `d930e295d93a51e242efaa477c996ba6e83d16632f394674cc1f301c1fc98f29`.
- `development-errors.json`: SHA256 `7c47ec88b05a5ba9a60ab6d7830112f1fe05093b5fe997461f9559e4f4a8d8ec`; its recorded manifest digest matches the exact manifest above.
