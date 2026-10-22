# Second and final hypothesis: fixed channel dropout during TCN training

The first robustness experiment found development benefit from ordinary mean
correction but did not support the added bound's proposed movement-mixture
advantage. The prespecified decision retained frozen normalization. This second
hypothesis tests a different, fixed training intervention without reopening that
parameter search or any sealed recording.

**Observation.** The original TCN has unstable S2 performance across seeds, from
-0.153394 to 0.012743, despite positive S3/S4 performance. This does not establish
underfitting, overfitting, missing channels, or a biological recording mechanism.
Declining training loss alone would not establish underfitting.

**Mechanism hypothesis.** Training with some observation channels replaced by
their source mean may reduce dependence on individual channels and improve
performance when observations differ across days. Unstable or missing channels
are a hypothesis here, not an identified explanation for these H1 recordings.

**Prediction.** One fixed channel-dropout setting should improve seed-averaged
forward-day R² without materially harming any seed-averaged day. The inference
architecture and causal behavior remain exactly those of the original TCN.

**Discriminating experiment.** `configs/tcn-dropout-v1.json` changes training
only: independently drop each channel with probability 0.1 for each training
chunk. The sampled mask remains fixed across the entire chunk, including its
14-bin burn-in. Missing standardized channels become zero, the frozen source
mean; surviving channels are not rescaled. Masks use the seeded Torch RNG.
Inference uses all observed channels and has no dropout.

All other model/training settings match the original fixed TCN: three causal
32-channel layers, 20 epochs, Adam learning rate 0.001, no weight decay, 256 scored
positions per chunk, 14 preceding context bins, 16 chunks per batch, and final
epoch weights. Seeds 0, 1, and 2 each run S0–S1 → S2, S0–S2 → S3, and S0–S3 → S4.
Only this probability is tested; no best seed or epoch is selected. The default
`fit_temporal` path uses dropout zero, returns its input unchanged, and consumes
no extra random draws, preserving the original training path.

**Decision rule, fixed before execution.** Retain augmentation only if the mean
of all three seed-level development means gains at least 0.005 absolute R² and
no day's mean over those three seeds loses more than 0.01 relative to the original
TCN. Both conditions must pass. Keep all nine seed/day values, paired differences,
and failures. If it fails, retain the original TCN and stop augmentation search.
These observations remain development selection, not final evidence.

```sh
PYTHONPATH=src .venv/bin/python -m neural_decoder.train_augmented \
  --index data/neural_h1/prepared/index.json \
  --baseline-run results/neural-h1/tcn-dev-v1 \
  --config neural_experiments/configs/tcn-dropout-v1.json \
  --output results/neural-h1/tcn-dropout-dev-v1
```

Fresh output retains source and configuration snapshots, data/model identities,
losses, full predictions, per-day/per-output scores, complete seed summaries,
and the explicit retention decision. The original results remain immutable.
An independent whole-file inference and sklearn score audit checks saved numeric
weights after training. No native compilation, timed runtime comparison, or S5+
sample access is required by this experiment.

## Completed development comparison

The single configured experiment completed on October 22, 2026 in
`results/neural-h1/tcn-dropout-dev-v1`. It **passed both prespecified retention
criteria**. The mean over all seed-level means increased from 0.160722 to
**0.172153**, an absolute R² gain of **0.011431**. Each seed-averaged day improved:
S2 +0.010648, S3 +0.005737, and S4 +0.017909. The decision retains the
`tcn-dropout-v1` recipe for the later freeze review; it does not select a best seed.

| Seed | S2 R² | S3 R² | S4 R² | Equal-day mean R² | Mean difference vs original |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.063407 | 0.311225 | 0.315891 | 0.187903 | +0.038768 |
| 1 | -0.095459 | 0.235801 | 0.278058 | 0.139467 | +0.002401 |
| 2 | 0.013870 | 0.237938 | 0.315457 | 0.189089 | -0.006876 |

The population standard deviation across the three augmented seed means is
0.023118. This is seed variation, not a confidence interval or an estimate based
on nine independent biological days. Individual comparisons still show harm:
seed 1 loses 0.059171 on S2, seed 2 loses 0.029548 on S3, and seed 0 loses
0.000954 on S4. Those observations remain in `comparison.json`; they are not
removed by the positive seed-averaged decision. Two of the three augmented S2
scores remain negative. The results therefore support this fixed development
choice while preserving substantial session/seed uncertainty.

Independent inference audit checked all nine exported models and 18 full traces
using functional whole-file Torch convolution from numeric weights, without
the production incremental cache implementation. Maximum absolute prediction
difference was **3.3528e-08**, and maximum absolute resulting day R² difference
was **1.2144e-07**. Independent float64 sklearn scoring matched the saved
day/output scores within 1e-12. Training-only normalizers and source/configuration,
input, model, and prediction hashes were checked. The result and exact audit
source are retained as `independent-audit.json` and
`independent-audit-source.py` beside the run. No additional training was performed
for this audit.

This experiment does not show that real channels went missing or establish the
reason the original decoder failed. Synthetic dropout is constant across a
training chunk, which need not resemble the time course of recording changes.
It changes training stochasticity as well as input availability. The measured
effect is a modest improvement under the fixed, matched development comparison.
The source remains an exploratory snapshot; root owns the committed recipe
freeze and untouched assessments. No second dropout rate, extra epoch count,
architecture, or third hypothesis was searched.
