# Prespecified bounded mean-correction experiment

This is the first G3 mechanism hypothesis described in NOTEBOOK.md. The selected
ridge weights from `ridge-dev-v1/ridge-h1-lambda0.1` remain fixed separately for
each forward fold. No new model is fitted, and no S5 or later-day array is used.
The fixed configuration is `configs/adaptation-v1.json`, written before the
diagnostic scores were computed. The three caps are one declared comparison;
an unsuccessful result does not authorize a new search.

## Causal equations and state

Let `f_t` be the existing 12-tap causal FIR output, `mu` and `sigma` the frozen
training channel mean and SD, and `W,b` the existing history-one ridge weights.
Initialize EMA deviation `m_0=0`. With `alpha=1-exp(-0.02/30)`, each step performs:

```
c_t = 0                                      (frozen)
c_t = m_t                                    (ordinary EMA)
c_t = clip(m_t, -cap*sigma, cap*sigma)         (bounded EMA)
y_hat_t = ((f_t - mu - c_t) / sigma) @ W + b
m_(t+1) = (1-alpha)*m_t + alpha*(f_t - mu)     (EMA modes only)
```

Caps are 0.25, 0.5, and 1.0 training SD. Variance is fixed. Applied correction is
bounded; the underlying EMA is identical between ordinary and bounded versions.
Thus the correction is anchored to the training mean. Clipping bounds magnitude,
while the 30-second time constant controls EMA response under ordinary inputs;
it does not guarantee a small update under arbitrarily large input values.

Current neural counts enter the ordinary causal FIR state for the current
prediction. The prediction uses the previous adaptation state, then all EMA
update work completes before `step` returns. Trials and score masks are not
predictor inputs. Each provider file resets FIR, history, and EMA state. Numeric
state has fixed size; the Python implementation still allocates scratch arrays.
No allocation-free or timing claim is made.

## Locked diagnostics

The real condition replays unmodified S2, S3, and S4 files. The observation
condition adds `+0.5 * training_filtered_channel_SD` to every raw channel at every
bin before filtering. These nonnegative real-valued synthetic inputs represent
a controlled arithmetic offset, not a physiological count-generating mechanism.
The fixed FIR starts with zero raw history, so the offset ramps through the
filter at file startup. Targets and evaluation masks stay unchanged.

The movement-mixture diagnostic selects one contiguous 60-second/3,000-bin
interval per file maximizing the signed mean of target g3 (output index 6),
using all rows for selection. Ties choose the earliest start. Every file must
contain at least 3,000 bins; otherwise the run fails. The selected interval
retains its original score mask and row order. It starts from reset without
preceding context, and intervals are never stitched together. Selection is a
retrospective evaluator-only operation: the predictor receives only the chosen
neural sequence. This construction combines changed movement mixture with a
fresh stream start, which limits interpretation. It is not an untouched final
recording assessment.

## Decision and evidence

Every method is compared with frozen centering within the same condition and
same files/windows. A bounded candidate must meet all four conditions:

- Real mean day R² gain at least 0.005.
- No real day harmed by more than 0.01 absolute R².
- Positive mean R² gain in the synthetic observation-offset condition.
- Strictly smaller worst-day harm in the mixture condition than ordinary EMA.
  Harm is the negative paired difference versus frozen, floored at zero.

If multiple caps qualify, prefer the largest real mean gain, then the smaller
cap. If none qualifies, retain frozen normalization and abandon this tested
mean-correction hypothesis. These are development decision thresholds, not
biological or clinical requirements.

The fresh result directory retains source/configuration/input/model identities,
the selected diagnostic intervals, all full predictions, final state artifacts,
per-day and per-output R², paired differences, days harmed, worst harm, and the
decomposition of residual SSE into mean bias and centered error. The real frozen
trace must exactly reproduce the retained ridge baseline before the study can
complete. A source snapshot binds exploratory execution; root owns any later
freeze of committed source and sealed assessment.

```sh
PYTHONPATH=src .venv/bin/python -m neural_decoder.run_adaptation \
  --index data/neural_h1/prepared/index.json \
  --baseline-run results/neural-h1/ridge-dev-v1 \
  --config neural_experiments/configs/adaptation-v1.json \
  --output results/neural-h1/adaptation-dev-v1
```

## Development result: bounded rationale not supported

The locked experiment completed on October 22, 2026. All predictions and final
states are retained in `results/neural-h1/adaptation-dev-v1`. The unmodified
frozen traces exactly matched the original selected ridge predictions. The
prespecified decision is **retain frozen normalization**: no cap demonstrated
less mixture-stress harm than ordinary EMA, because ordinary EMA harmed no day
in that diagnostic. The mean-correction arithmetic improved these development
observations, but the reason for adding the proposed bound was not supported.

An independent arithmetic recomputation with SciPy's delayed EMA filter and a
batch affine calculation checked all 90 prediction traces, their final EMA
states, and all 45 condition/method/day scores. Maximum absolute prediction
difference was 3.8164e-17; independent float64 sklearn scores agreed within
1e-12. Hashes were checked, and the script/result are retained as
`arithmetic-audit-source.py` and `arithmetic-audit.json` in the run directory.

| Method | Real mean R² | Offset mean R² | Mixture-window mean R² |
|---|---:|---:|---:|
| Frozen | 0.052625 | 0.055136 | 0.064840 |
| Ordinary EMA | 0.142009 | 0.141384 | 0.139038 |
| Bounded, cap 0.25 SD | 0.086526 | 0.074028 | 0.095994 |
| Bounded, cap 0.5 SD | 0.104685 | 0.089816 | 0.110325 |
| Bounded, cap 1.0 SD | 0.123421 | 0.113285 | 0.126740 |

| Method | Real S2 R² | Real S3 R² | Real S4 R² | Real days harmed vs frozen |
|---|---:|---:|---:|---:|
| Frozen | -0.072894 | 0.099554 | 0.131214 | 0/3 |
| Ordinary EMA | 0.110719 | 0.138370 | 0.176937 | 0/3 |
| Bounded, cap 0.25 SD | -0.006671 | 0.108405 | 0.157843 | 0/3 |
| Bounded, cap 0.5 SD | 0.027357 | 0.119229 | 0.167470 | 0/3 |
| Bounded, cap 1.0 SD | 0.068388 | 0.130505 | 0.171370 | 0/3 |

Every cap exceeded the required real mean gain of 0.005, met the real worst-day
harm ceiling, and improved the mean offset-condition score. All caps and ordinary
EMA improved every mixture-window day's score relative to its frozen comparator.
Thus their worst-day mixture harms all equal zero, failing the locked **strictly
less harm** criterion. In the offset condition only cap 0.25 harmed one day,
S4, by 0.000909 absolute R²; all other adaptive condition/day comparisons improved.
The machine-readable decision retains each passed/failed criterion rather than
recasting a failed guard as a success.

Ordinary EMA achieved the highest mean in all three conditions. That useful
development observation is reported without promoting it under a decision rule
written for the bounded candidate. The finite mixture test did not expose the
anticipated recentering failure and therefore does not establish that ordinary
EMA is safe under arbitrary movement-mixture change. No new stress magnitude,
window rule, time constant, or cap was searched after these results.

The imposed positive offset slightly improved the frozen mean relative to the
unmodified condition (0.055136 versus 0.052625), although individual days moved
in different directions. It is therefore inappropriate to describe this stress
as a uniformly damaging recording drift that adaptation repaired. Neither the
offset experiment nor the mean residual reduction identifies a biological
recording mechanism. All three conditions remain development diagnostics, and
the label-selected windows include fresh-start effects. These results do not
constitute a sealed-day assessment, official FALCON score, or runtime measurement.
