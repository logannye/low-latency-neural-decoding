# H1 V2 development accuracy study

This study compares temporal context, the loss assigned to the seven output
coordinates, and supervised readout calibration on H1 development days S0–S4.
It is a development study in one participant. It does not reopen the published
S5–S12 assessment, establish a new untouched-test improvement, or test transfer
between participants. All original production modules and V1 reports remain
unchanged.

## Locked protocol

The pre-execution plan is [accuracy_plan.json](accuracy_plan.json), SHA256
`db16b388f99a4c1105c0ef4aac3bd6dda2ee2f11d67455ec187be1325fbce8c3`.
The executed implementation is retained privately with SHA256
`3cba6d210143409f4b699c8a1f646c67c63586c9fa91191187fe559a0c281a29`.
The run creates a plan-lock receipt before fitting and a source snapshot with
the Git revision and individual source-file hashes. The private run directory
is `results/neural-v2/accuracy-v1`; public results contain aggregates and hashes,
without model weights, calibration coefficients, or sample arrays.

Four arms share width 32, one causal kernel-3 convolution and ReLU per layer,
a linear seven-output readout, Adam at 0.001, and 0.1 channel dropout. The
dropout mask is constant within a chunk and replaces a standardized channel
with its source-training mean. RF15 uses dilations 1, 2, 4 and RF63 uses
1, 2, 4, 8, 16. The windows contain 15 and 63 samples at 20 ms per sample;
their oldest samples precede the current sample by 280 and 1240 ms. Neither
architecture uses future samples.

Each receptive field is paired with standardized MSE or source-unit MSE.
The latter weights each squared standardized residual by the corresponding
source-training target variance divided by the mean target variance. The
normalization preserves a comparable loss scale while directing the fit
toward squared error in the original output units. The weights use no
evaluation labels. This need not improve every coordinate's R².

Input statistics use source-training neural rows; target statistics use only
source-training rows bearing labels. Chunks contain 256 potential loss
positions plus RF−1 real preceding samples from the same file. All four arms
exclude the first 62 bins of each training file from loss, which holds loss
positions constant across receptive fields. Right padding contributes no
loss. Evaluation predicts every bin from file startup, including unscored
bins, with zero activation history at reset. The common 62-bin training
exclusion and nested epoch choice make the V2 baseline a new controlled
comparison, rather than a numerical reproduction of the published V1 fit.

## Schedule, comparisons, and calibration

Before scoring S2–S4, each arm trains for 40 epochs with seed 0 on
S0_set_1, S0_set_2, S1_set_1, and S1_set_2. Checkpoints at epochs 20 and 40
are scored on S1_set_3. One global schedule selects 40 only if the four-arm
mean gain is at least 0.005 and no arm loses more than 0.01; otherwise all
subsequent fits use 20. The decision is written and hashed before loading
the outer evaluation arrays. This is a bounded convergence check, not a
search over optimizers, widths, learning rates, or held-out days.

The main comparison uses seeds 0, 1, and 2 on chronological folds
S0–S1→S2, S0–S2→S3, and S0–S3→S4. Reported aggregate R² gives each day and
seed equal weight. Each day's seven-output R² is variance weighted, with
the original scoring mask and full-file predictions.

The exposure diagnostic fits only RF15 with standardized loss. For each
of S2, S3, and S4, the same-day condition uses the first 25%, 50%, or 100%
of the first file. Its comparison uses the same number of raw bins from
the chronological tail of earlier-day files. After excluding the first
62 bins of every selected segment, both conditions retain exactly the
smaller available count of scored labels. Both evaluate the identical,
disjoint last file of the target day. Source boundaries remain resets.
The neural duration and label count are matched, but behavior coverage,
recency within source data, and the number of resets may differ. These
comparisons diagnose drift and exposure; they do not identify a single
causal biological explanation for a performance gap.

Every main model also receives an unadapted, intercept-only, and ridge
delta-head comparison at 0, 10, and 30 nominal seconds per evaluation file.
Only labels with the original scoring mask inside that fixed prefix are
available to fitting. A 32-dimensional final-layer activation is standardized
using source-training neural statistics. Ridge fits a residual correction
around the original head with mean squared residual loss, penalty 1.0 on
the coefficient matrix, and an unpenalized intercept. There is no penalty
search and no backbone update. All methods are scored on the same suffix
beginning at bin 1500, after 30 seconds. Prefix predictions remain unadapted
until the declared fitting boundary. Zero-budget methods coincide.

## Candidate selection and streaming

A candidate arm is eligible only if its paired mean gain over RF15 with
standardized loss is at least 0.005 across all nine day/seed comparisons,
and no seed-averaged day loses more than 0.01. Selection uses highest mean
R² among eligible arms, then fewer parameters, then plan order. If none
qualifies, the baseline is retained. The runtime model is always seed 0
trained through S3 for S4 evaluation in the selected arm; seeds are never
ranked for deployment selection. This remains development-based selection.

`accuracy.load_model(path)` returns the causal `TemporalCore`, four affine
transforms, and architecture metadata. `accuracy.predict(model, transforms,
counts)` predicts a complete file. `accuracy.CausalStream(model, transforms)`
provides `step`, `predict_chunk`, and `reset`. The stream consumes neural
counts only and maintains activation history for every dilation, including
the correct zero latent history before file startup. Numeric NPZ exports
use version 1, explicit dimensions/dilations, float32 transforms, and exact
float32 state tensor shapes; loading disables pickle.

The focused synthetic tests cover the unchanged RF15 architecture, RF63
causality, loss-position and padding alignment, full/chunk/step parity,
reset, train/export/replay, matched exposure, prefix-only calibration, and
the deterministic selection rule. The real run also compares full-file
step predictions against batched predictions for both S2 files for seed 0
of each arm. These correctness checks do not establish latency. Training
uses the existing CPU environment with Torch and BLAS limited to two threads.

## Results

All 94 fits completed in 552.0 seconds on two CPU threads. The nested
validation gain from epoch 20 to 40 averaged 0.051989, with every arm
improving; the fixed rule therefore selected 40 epochs. All four-arm results
below use that same schedule. The complete aggregate, individual day/seed
scores, per-output scores, prefix counts, loss-curve identities, and private
artifact hashes are in [v2-accuracy.json](../results/v2-accuracy.json).

| Arm | Mean development R² | Seed-mean SD | S2 | S3 | S4 | Paired gain vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF15, standardized | 0.159170 | 0.027916 | −0.019970 | 0.265965 | 0.231514 | 0 |
| RF63, standardized | 0.041082 | 0.096616 | −0.045850 | 0.338664 | −0.169568 | −0.118088 |
| RF15, source units | 0.146591 | 0.011708 | 0.001544 | 0.244287 | 0.193941 | −0.012579 |
| RF63, source units | −0.023503 | 0.064363 | 0.135201 | 0.310860 | −0.516570 | −0.182673 |

Day columns average the three seeds; SD is the population SD of the three
seed means, not uncertainty across independent participants. None of the
three changes met the prespecified improvement gate. Longer context helped
some folds but produced severe S4 regressions. Loss reweighting alone also
failed to improve the aggregate. The selected unadapted candidate remains
RF15 with standardized loss, seed 0/S4, 23,367 parameters, model SHA256
`81334d4f8c838ffa96986f4befbdd993dce878edd6f9da538d9946cb00f2995f`.
These results do not establish that longer histories are intrinsically
unhelpful; they reject these particular fixed recipes under this protocol.

Matched exposure made the same-day versus earlier-day difference clearer:

| Fraction of first same-day file | Same-day training R² | Earlier-day training R² | Matched eligible labels per fit |
| --- | ---: | ---: | ---: |
| 25% | 0.030910 | −0.305762 | 2,366–2,834 |
| 50% | 0.093846 | −0.251247 | 4,816–5,732 |
| 100% | 0.216539 | −0.110433 | 9,704–11,609 |

Each score averages the same three held-out last files and three seeds.
Same-day training won 26 of 27 paired day/fraction/seed comparisons; the
exception lost 0.016219 R². More same-day exposure improved the aggregate
at every fraction, while equal amounts of earlier-day data performed much
worse. This supports distribution shift as a practical accuracy constraint,
subject to the behavior and reset-count differences described above. These
scores cannot be compared directly with the main fold averages: the
diagnostic uses only the last file and much less training data.

The common-suffix calibration comparison found the strongest positive
signal. The suffix always begins after 30 nominal seconds, including for
unadapted and 10-second methods:

| Backbone/loss | Unadapted | Intercept 10 s | Ridge head 10 s | Intercept 30 s | Ridge head 30 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF15, standardized | 0.174224 | 0.199948 | 0.142394 | 0.219789 | 0.298100 |
| RF63, standardized | 0.060279 | 0.093666 | 0.127718 | 0.135764 | 0.299072 |
| RF15, source units | 0.162178 | 0.172016 | 0.085321 | 0.201430 | 0.273007 |
| RF63, source units | 0.003283 | 0.030495 | 0.185109 | 0.084399 | 0.323194 |

For the selected RF15 baseline, 30-second ridge head calibration improved
all nine day/seed comparisons over both the unadapted model and the
30-second intercept correction. Its mean gain was 0.123876 over unadapted
and 0.078311 over intercept-only. The smallest gains were 0.022514 and
0.006521 respectively. In contrast, its 10-second ridge correction improved
only four of nine comparisons and reduced mean R² by 0.031830. The 10-second
prefix contained 416–487 scored labels per file; the 30-second prefix
contained 1,407–1,481. Budgets refer to nominal wall-time prefixes, not that
many seconds of fully scored labels.

The RF63/source-unit model has the highest descriptive 30-second calibrated
mean, but one of its nine comparisons still loses relative to its own
unadapted model. Calibration outcomes were not the locked unadapted
selection criterion, so that model is not substituted as the runtime
candidate. These comparisons motivate further development of supervised
adaptation; they do not establish zero-shot robustness or cross-person
transfer, and they do not replace untouched-test evaluation.

All 1,689 retained artifacts passed their hash checks. Independent sklearn
recomputation matched all 422 score records exactly, including per-output
R². The audit also verified common suffix counts, prefix label counts,
unchanged predictions before fitting, zero-budget identity, diagnostic
training/evaluation file exclusion, and the candidate gate. Real S2
stream-versus-batch checks across all four arms had maximum absolute error
4.47×10⁻⁸. The audit receipt and source hash accompany the public aggregate;
full traces remain private. The published independent auditor is
[audit_accuracy.py](audit_accuracy.py), SHA256
`67a0e5cd2fcd6e1752009b092deb9f952ea584a7aeed006089ffb1c7edc5161a`.
It was rerun against the unchanged study, with a fresh receipt. The earlier
private auditor's distinct hash and receipt are also retained. A synthetic
fixture verifies that even a false R² with internally matching artifact
hashes is rejected by independent score recomputation.
The retained plan is resolved by its frozen manifest hash, allowing a
reproduction to use a different plan filename without changing its contents.

## Frozen calibrated runtime artifact

[compose_head.py](compose_head.py) folds the already-fit 30-second ridge
correction for the baseline's seed 0/S4 model and S4_set_1 into its linear
head. It performs no fitting or search. For final-layer activation `z`,
the retained correction is `((z − latent_mean) / latent_scale) @ C + b`.
Composition adds `(C / latent_scale / target_scale).T` to the standardized
head weight and `(b − (latent_mean / latent_scale) @ C) / target_scale` to
its bias. The backbone, affine normalization, parameter count, and receptive
field remain unchanged; the new head is stored in float32.

The resulting private model is
`results/neural-v2/composed-head-v1/model.npz`, SHA256
`0f50648a5f65b196308f5bba083cc1640dca085c677c06b0cd765a8fae38f4c5`.
Its receipt binds the parent checkpoint, calibration coefficients, latent
statistics, canonical feature file, and retained predictions. The model
was saved before suffix targets were opened for evaluation. On the 10,214
scored suffix bins of S4_set_1, its maximum prediction difference was
1.0044×10⁻⁸ and its R² differed by 8.96×10⁻¹⁰ from the retained calibrated
trace. The same single-file suffix R² was 0.234764 in both cases at six
decimals. This single-file score is distinct from the nine-comparison
calibration mean above.

Replaying this calibrated artifact from the first bin measures runtime
only: its weights incorporate labels from the first 30 seconds of that
file. Scientific accuracy claims begin after that prefix. The composition
is not a zero-label or zero-shot result. Two additional synthetic tests
verify the algebra, preserved backbone/receptive field, export/reload,
and causal-stream parity. The combined focused suite passed all 12 tests.

To reproduce the independent score audit with locally available private
artifacts, choose a fresh receipt path:

```sh
PYTHONPATH=src .venv/bin/python neural_experiments/v2/audit_accuracy.py \
  --run results/neural-v2/accuracy-v1 \
  --index data/neural_h1/prepared/index.json \
  --output results/neural-v2/accuracy-audit-reproduction.json
```

The study and composition CLIs similarly require fresh output directories;
they preserve existing runs and receipts.
