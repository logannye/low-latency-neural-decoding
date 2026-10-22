# T3 exposed-window diagnosis (post hoc)

This is exploratory development evidence. It reuses all eight T3 sessions already
exposed by V2/V3 and only their prepared first 600 seconds. No raw files, retained
tails, newly acquired observations, or future V4 outcomes were opened. The two
former V3 assessment sessions are now development data; their subgroup values are
descriptive, not a held-out test. The diagnostic JSON pins both indexes, every
feature/target artifact, the prior public V3 receipt, and the executed script and
frozen V3 model source. Sample arrays and model coefficients are not exported.

`diagnosis.py` executes 16 fixed diagnostic recipes plus two constant controls
at 30- and 120-second prefixes. Every comparison below scores the same
`[302,600)` suffix and averages sessions equally. All neural filters remain
causal and reset by acquisition block; fitting uses only the permitted prefix,
with the V3 62-bin per-block burn and mean-loss ridge convention. Calibration
training scores are optimistic fit diagnostics. Per-block R² values in the JSON
can be unstable for short, nearly constant-direction end fragments; no block
R² is averaged into the session-level comparison.

## Findings

- TX4.5 is very sparse: only 0.15–0.95% of prefix120 count entries are nonzero,
  and 97.9–100% of channels have filtered prefix standard deviation below the
  existing 0.1 floor. Matched TX-only training R² averages 0.035, versus 0.335
  with combined TX/SBP. This is consistent with limited available count signal;
  it does not establish that all released threshold choices are uninformative.
- SBP dominates the combined fit. At 120 seconds, the combined and SBP-only
  suffix means are −0.3513 and −0.3519. Removing TX barely changes the failure.
  Calibration fit is positive while later performance is negative, including
  some held-out time within the initial acquisition block.
- Day62 provides a particularly clear block-shift example. The combined model
  scores −0.096 after calibration within block1 and −0.891 in block2. Its suffix
  mean-bias contribution is 43.2% of squared error. SBP mean displacement reaches
  1.10 prefix standard deviations RMS across channels. Day79 reaches 18.52, but
  its suffix score is less poor; feature shift magnitude alone is not a causal
  explanation for every performance difference.
- Common-mode removal scarcely helps. A past-only per-channel 5-second running
  baseline reduces SBP error substantially on day62, but neither this nor the
  combined common-mode/running-baseline variant gives positive average R².
  Drift correction reduces some damage without establishing useful signal.
- Calibration has much less directional diversity than its bin count implies.
  The first 30 seconds contain only 1–4 contiguous scoring runs; the first 120
  contain 2–11. These are scoring runs, not an asserted count of distinct trials.
  Prefix120 mean-direction norm ranges 0.113–0.498; normalized octant entropy
  ranges 0.671–0.919. Adjacent direction dot products are about 0.996–1.000 and
  one-second values 0.787–0.978. Random bin splits would be misleading.

| Recipe | Common-suffix R², 30 s | Common-suffix R², 120 s |
|---|---:|---:|
| Zero output | −0.0401 | −0.0401 |
| Prefix mean | −0.2357 | −0.1353 |
| Combined, τ=.2 s, λ=1 | −0.4784 | −0.3513 |
| TX only, τ=.2 s, λ=1 | −0.2335 | −0.1352 |
| SBP only, τ=.2 s, λ=1 | −0.4812 | −0.3519 |
| Combined, λ=10 | −0.2610 | −0.1496 |
| Combined, λ=100 | −0.2362 | −0.1359 |
| Combined, τ=.4 s | −0.5097 | −0.4612 |
| Combined, slow 30 s subtraction | −0.9989 | −0.3091 |
| SBP current-bin channel-mean removal | −0.4792 | −0.3337 |
| SBP past-only 5 s recenter | −0.3720 | −0.2056 |
| SBP both corrections | −0.3693 | −0.2062 |
| TX with fitted intercept set to zero | −0.0450 | −0.0366 |

The zero-intercept TX result is only 0.0035 R² above zero at 120 seconds. It is
primarily a conservative reduction in error, not evidence of a substantial
accuracy repair. Lower regularization (.1) and longer smoothing (.4 seconds)
increase combined-model calibration fit while worsening suffix accuracy. No
positive-mean T3 decoder emerged from these pilots.

## Follow-up: is the TX standard-deviation floor suppressing signal?

The separate `diagnosis_floor.py` tests six specified alternatives plus the
existing TX-only τ=.4, floor=.1, λ=1 reference, using exactly the same eight
sessions, masks and suffix. Its complete aggregate is `t3-floor-pilot.json`.
The floor does suppress calibration fit, but relaxing it did not recover useful
out-of-prefix information:

| TX recipe at 120 s | Training R² | Common-suffix R² |
|---|---:|---:|
| Floor .1, λ=1, free intercept | 0.0410 | −0.1363 |
| Floor .01, λ=1, free intercept | 0.2650 | −0.1748 |
| Floor .001, λ=1, free intercept | 0.3907 | −0.2367 |
| Floor .01, λ=.1, free intercept | 0.4675 | −0.3375 |
| Floor .001, λ=.1, free intercept | 0.5211 | −0.4329 |
| Floor .01, λ=1, intercept ×.25 | 0.1863 | −0.0771 |
| Floor .001, λ=1, intercept ×0 | 0.2509 | −0.1401 |

The 30-second pilots show the same overfitting pattern: training R² can reach
0.857 while suffix R² falls to −0.499. A lower floor is not supported as the
standalone TX4.5 repair. Lower-threshold TX3.0 is a distinct feature hypothesis
evaluated below.

## Follow-up: released TX3.0 counts

The authorized V4 development export adds the released `tx_3` threshold channel
on the same exposed windows. `t3-tx3-floor-pilot.json` repeats the identical seven
TX scale recipes at τ=.4 seconds with `x_tx3`. Full-window count rates are now
17.94–20.35 Hz per channel and 30.1–33.3% of entries are nonzero. More threshold
events are available, but the tested fits still overfit:

| TX3.0 recipe at 120 s | Training R² | Common-suffix R² |
|---|---:|---:|
| Floor .1, λ=1, free intercept | 0.3739 | −0.1979 |
| Floor .01, λ=1, free intercept | 0.4097 | −0.2206 |
| Floor .01, λ=.1, free intercept | 0.5263 | −0.3936 |
| Floor .01, λ=1, intercept ×.25 | 0.3316 | −0.1335 |
| Floor .001, λ=1, intercept ×0 | 0.2711 | −0.1337 |

At 30 seconds the reference training/suffix pair is 0.7463/−0.3049. The .01 and
.001 floor versions coincide here, since the TX3.0 channel variation exceeds
those small floors. These pilots do not test all possible thresholds or prove
neural information is absent; they do weaken a simple threshold-sparsity fix.
The post-fit intercept multipliers in this exploratory table are distinct from
the exact zero-intercept constrained ridge fit implemented for the finite V4
candidate set. No fresh V4 outcome has been inspected.

## Bounded directions worth testing prospectively

1. Keep any further count comparison finite and conservative: .2/.4-second
   causal smoothing, strong ridge, and an exactly constrained zero-intercept
   control. Lower TX thresholds supply more events, but their simple daily
   fits have not supplied useful suffix accuracy in these exposed pilots.
2. Treat calibration as a direction-coverage problem as well as a duration
   budget. A strong zero-centered intercept prior and conservative output can
   limit harm from a handful of imbalanced runs. More correlated bins alone
   do not supply broad calibration coverage.
3. If SBP remains in a model, retain an explicit causal per-channel drift
   correction or a low-complexity fallback; a per-sample common offset removal
   alone did not address the failure. This should compete against TX-only and
   zero on fresh locked data. A larger temporal network is not supported by
   the current low-signal and overfitting evidence.

Reproduce from the repository root using the existing environment. The CLIs
require explicit prepared indexes and a new output path, reject overwrite, and
read no raw files. These historical first600s windows are now exposed development
data. The current V4 assessment indexes must not be supplied.

```sh
export PYTHONPATH=.:src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m neural_experiments.v4.diagnosis \
  --development-index data/neural_v3/development/index.json \
  --assessment-index data/neural_v3/assessment/index.json \
  --prior-results neural_experiments/results/v3-results.json \
  --output work/reproduced-t3-diagnosis.json
python -m neural_experiments.v4.diagnosis_floor \
  --development-index data/neural_v3/development/index.json \
  --assessment-index data/neural_v3/assessment/index.json \
  --indices data/neural_v3/development/index.json data/neural_v3/assessment/index.json \
  --field x_tx --output work/reproduced-t3-floor45.json
python -m neural_experiments.v4.diagnosis_floor \
  --development-index data/neural_v3/development/index.json \
  --assessment-index data/neural_v3/assessment/index.json \
  --indices data/neural_v4/development/index.json \
  --field x_tx3 --output work/reproduced-t3-floor3.json
```

The [public aggregate bundle](../results/v4-diagnosis.json) preserves all three
original executed pilot receipts, their original source hashes, all negative
configurations, and fresh reproduction identities from the published CLIs.
The public CLI refactor has a new source identity; original receipts have not
been relabeled as having used that newer source. Numerical agreement is checked
apart from source identity and elapsed duration. The baseline reproduces all
four published T3 V3 selected120 partition scores within 1.3×10⁻¹⁵ R².
These diagnostics do not modify any V3 artifact or conclusion.
