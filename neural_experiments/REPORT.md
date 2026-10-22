# H1 development study: first results

October 22, 2026. This report covers G1 (data and baselines) and G2 (one compact
causal model). G3 adaptation, G4 sustained runtime measurements, and G5 sealed
assessment remain unfinished. These are **local development results on public
FALCON calibration recordings**, not official FALCON scores.

A 23,367-parameter causal TCN improves the mean development R² from **0.0526**
(selected ridge) to **0.1607** averaged over three seeds, an absolute difference of
**0.1081**. S2 remains unstable and frequently scores below the training-mean
baseline. No seed was selected for a final claim. This result supports further
investigation; it does not establish cross-day robustness on untouched data.

## What was evaluated

FALCON H1 provides one participant, 176 channel rows, 20 ms bins, and seven outputs
labeled `tx,ty,tz,rx,g1,g2,g3`, with units recorded as `arbitrary`. A row ID is not
an independently verified electrode identity. Individual grasp-label anatomy is
not inferred. The observations can change with recording conditions and movement
mixture; a shift in neural activity alone does not identify either cause.

The forward folds train on S0-S1 and evaluate S2, train on S0-S2 and evaluate S3,
and train on S0-S3 and evaluate S4. The 11 development recordings contain 112,488
bins, or 37.50 nominal minutes, of which 109,603 bins are scored. Evaluation days
S2/S3/S4 contain 22,790/21,746/23,368 scored bins. These numbers count bins and do
not claim independent observations. The provider's deidentified date labels imply
18 days between S0 and S4; across the entire metadata inventory, S0-S12 span 39 days.
The nominal 1925 year is a provider label, not an assertion about actual acquisition.

All 40 assets (102,258,934 bytes) have provider and local matching SHA256 digests.
Only development arrays were exported. S5 and all S6-S12 neural/behavioral sample
arrays remain sealed to development; only their schema metadata was inspected.
Every development minival file is an exact behavioral/mask/trial prefix of its
calibration file, so minival is excluded from independent evidence. Full identities,
terms, and timing caveats are in [the data audit](data/README.md).

## Development results

The primary metric concatenates files within each day, applies the official mask,
computes variance-weighted R², then equally averages days. Negative scores remain
negative. Per-output scores, sample counts, and the official population standard
deviation are retained in [development-results.json](development-results.json).

| Model | S2 | S3 | S4 | Mean over days |
|---|---:|---:|---:|---:|
| Training target mean | -0.000053 | -0.000028 | -0.000030 | -0.000037 |
| Selected ridge | -0.072894 | 0.099554 | 0.131214 | 0.052625 |
| TCN seed 0 | -0.153394 | 0.283955 | 0.316845 | 0.149135 |
| TCN seed 1 | -0.036288 | 0.216313 | 0.231170 | 0.137065 |
| TCN seed 2 | 0.012743 | 0.267486 | 0.307665 | 0.195964 |
| TCN seed average | -0.058979 | 0.255918 | 0.285226 | 0.160722 |

The TCN's three seed-level means are 0.149135, 0.137065, 0.195964. Their population
standard deviation is 0.025403; it describes initialization variation, not uncertainty
about a population of independent days. With three development days used for model
selection, no confidence interval or unbiased final-performance claim is made.

The ridge grid was fixed before scores: 12-tap causal exponential FIR, history 1
or 5, and regularization 0.001/0.01/0.1. Training-only feature mean/std and an
unpenalized intercept are fitted per fold. One filtered bin with penalty 0.1 wins
this grid. Five-bin history worsens S2. The constant predicts training masked-target
means. No target-conditioned still-period exclusions are used.

The TCN uses three single k=3, width 32, dilations 1/2/4 convolutions with ReLU and a
linear 7-output head, giving 15 bins (300 ms collection window) of causal support.
It consumes standardized raw counts, and target transforms are inverted before
scoring. The fixed recipe uses 20 Adam epochs at lr 0.001, batch 16 chunks, 256 loss
positions plus 14 preceding real bins, last-epoch checkpoints, and seeds 0/1/2.
First 14 bins/file are excluded only from training loss; evaluation scores all
provider-selected bins. A one-epoch timing-only pilot preceded the campaign. No
validation early stopping, new-day normalization, or best-seed selection occurred.

## Verification and implementation

The offline exporter exactly matches the pinned official loader's counts, targets,
masks, and trial-change flags for all 11 development files. H1's stored `rate=0.02`
is treated by the official loader as an interval; this behavior is preserved.
The exported count dtype is `uint32` rather than upstream `uint8`; observed development
counts are at most 8, so values are unchanged while overflow is guarded.

Features and scorer targets occupy separate files. Inference accepts counts and
approved feature metadata, never masks, labels, trials, or future bins. It predicts
on unscored bins, carries history across trials, and resets for each provider file.
The adapter matches the pinned H1 call semantics on synthetic fixtures; an official
container/submission has not been run.

Independent ridge reconstruction checked 21 models and 42 traces with maximum absolute
difference 5.90e-17, independently refit the selected three models, and recomputed all
scores. Independent TCN reconstruction loaded numeric weights and used whole-file
PyTorch convolutions to verify 9 models and 18 streaming traces. Maximum absolute
prediction difference was 2.79e-08; the largest resulting day-R² difference was 1.19e-07.
Independent sklearn scoring agrees with saved scores. All input/model/prediction
and retained-source hashes were checked. See the linked independent reviews.

Local validation: 131 tests passed, including existing CNN regression checks and
real development-cache validation; Ruff lint/format checks pass. Tests cover future
interventions, label/mask isolation, forged split identities, chunks, resets,
invalid channels/sequence IDs, scoring order, constant outputs, training/export,
streaming parity, bounded state, and corrupt artifacts. Linux hosted workflows are
tracked separately in GitHub Actions; their run status is not implied by this local
record. No Rust/C/assembly source was added or compiled.

Each exploratory run retains an exact source snapshot/diff rooted at historical
commit `6f0cfd932b41503fed3899211491bacfbfba95e7`, configuration, dependencies, input
identity, checkpoints, full predictions, losses, scores, and completion state.
That historical commit does not contain the new implementation. Later ingestion
guard fixes and formatting are tested in the publication tree; they are not
retroactively substituted for recorded run source. Final runtime/assessment
campaigns will require a frozen, committed source state.

The experiment ran on an Apple M4 Max, 128 GB RAM, macOS 26.2, Python 3.12, one CPU
compute thread. New study cache and run bundles occupy about 140 MiB, below the 5 GiB
budget. Installed prebuilt dependencies are shared with the existing environment;
there is no second dataset or native build tree. Training duration and short parity
traces are not runtime-tail benchmarks.

## First hypothesis and next gate

S2's selected ridge errors contain a larger mean-residual bias component: 18.19% of
its squared error, versus 4.82% on S3 and 5.58% on S4. Output `g3` contributes most of
that bias. However, S3's neural mean shift is larger than S2's, despite better
decoding. Offset correction is therefore a hypothesis, not an identified cause.

The next bounded study will compare frozen normalization, ordinary causal online
normalization, and one source-anchored bounded correction. It must use previous
adaptation parameters for prediction and update them only afterward. The [notebook](NOTEBOOK.md)
records the observation, mechanism, prediction, falsification experiment and keep/
abandon rule. Controlled observation changes and changed task mixtures must be
reported separately from real forward-day scores. S5-S12 stay sealed until recipe
freeze. No adaptation benefit, sustained speedup, deadline reliability, closed-loop
control, cross-participant transfer, or clinical benefit is claimed.

## Reproduce and inspect

[Reproduction commands](README.md), [protocol](PROTOCOL.md), [ridge review](BASELINE_REVIEW.md),
[TCN review](TCN_REVIEW.md), [TCN implementation contract](TCN.md), and
[scoring contract](SCORING.md) explain the evidence. Local raw/prepared data,
checkpoints, and sample-level predictions are not redistributed. Aggregate results,
source, manifests, and independent auditors are published for reproduction by users
who obtain the dataset under its provider terms.
