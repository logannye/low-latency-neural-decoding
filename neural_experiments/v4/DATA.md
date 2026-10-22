# V4 data roles and provenance

V4 separates two assessment domains. The new-session domain selects regular MAT
members 10–12 from the pinned archives for T2, T5, T6, T7, T8, T9, T10, and T11.
The T3 continuation domain uses the first two complete acquisition blocks that
start at or after the already exposed 600-second prefix in each of T3's eight
cached sessions. T3 has no unused session identities in this release. New time
blocks in known sessions must be reported separately from new sessions in the
other eight participants; neither is a new-person replication.

The [acquisition plan](acquisition-plan.json) was committed before acquisition,
with SHA256 `dadc1cce640900008087c9357527032fea85394b797fc21bb45fb02c3baf4fbe`.
Archive rank determines selection, not chronological order, duration, signal
quality, or decoding outcomes. Selected sessions are sorted by numeric
post-implant day only after selection. Historical training must use strictly
earlier days, never archive-rank order. Missing and unusable cases are retained
as dispositions without replacement.

## Completed preparation and validation

The development export is complete and passed full source-member, artifact-hash,
and array-schema validation: 80 previously exposed sessions, 2,261,767 bins.
Every session is capped at its first `min(600 seconds, duration)` of within-block
paired observations. Index SHA256:
`aa6bf303c746edde0fb4d0a3a937a0fb07afb849270f2c79093fc021a9c2c500`.
The original V2 and V3 inputs remain unchanged.

New-session acquisition completed on 2026-10-22 with all 24 requested members,
without missing ranks, failed participants, or substitutions. Exact transfer was
11,073,159,168 compressed bytes and retained raw storage was 2,617,107,571 bytes,
within the respective 18,000,000,000-byte and 12,000,000,000-byte limits. Three
workers shared response-byte accounting and reserved selected member sizes
before writing. Every prior regular member 1–9 was authenticated while
streaming; only members 10–12 were retained. Published whole-archive hashes
are not claimed verified when only a prefix was consumed. Signed transport
URLs stay private.

The new-session export passed full source-member, artifact-hash, and
array-schema validation. It contains 647,898 bins across 24 sessions and eight
people, or 3.59943 recorded feature hours. Index SHA256:
`76293562fd6e586e7d6f5506c41ebda97241c971e18984c22489b69ac29558cc`.
The fixed `[122,600)` score-window geometry totals 10,038.06 seconds before
scorer masks and any declared decoder burn-in; this differs from exported
feature duration.

The T3 continuation export is complete and passed full identity/schema and
label-isolation validation. It contains 482,114 causal-history feature bins and
172,182 bins in the fixed selected blocks, across eight sessions. Its index
SHA256 is `7f8908c54312814ce41b7e743042e85fb890a443ee267f8fed937ae66b27de69`.
The committed [full study plan](plan.json), SHA256
`f00419baf098ab5e42d6757714bd43cc5107579edf2a7c915c24a4f6ff7d8aba`, authorized
both fresh exports. The [public cohort receipt](../results/v4-cohort.json) binds
all three indexes, source members, prepared-artifact hashes, selected block
geometry, and acquisition accounting. All 112 exported session entries passed
an electrode-layout comparison with their participant-specific development
reference (electrode ID, array ID, x, and y).

Every fresh exporter requires an explicit full study-plan path and SHA256 and
checks that those exact plan bytes are committed before opening fresh arrays.
Preparation and schema validation compute no model scores. Source choices
passed independent audit and were published at commit `b400c8a` before the
locked assessment began. A reproduction must complete the same audit and
publication gate before opening fresh labels for assessment.

Metadata already establishes one duration shortage: new T6 day 1034 has only
113.90 recorded seconds. It cannot support the fixed evaluation window starting
at 122 seconds and is retained as unavailable, without substitution.

## Pure features and separated targets

The exact provider field name is `tx_3`, corresponding to the −3.0 robust-standard-
deviation voltage threshold. The [provider README](https://datadryad.org/dataset/doi:10.5061/dryad.x0k6djj1h)
also documents −3.5, −4, −4.5, −5, and −5.5 thresholds and spike-band power in µV².
V4 exports only the prespecified fields below; no threshold is substituted when
one is unavailable.

| Feature-file key | Type and shape | Preparation |
| --- | --- | --- |
| `x_tx3` | float32 `[T,C]` | Sum adjacent 10 ms `tx_3` bins |
| `x_tx45` | float32 `[T,C]` | Sum the same `tx_4_5` pairs |
| `x_sbp` | float32 `[T,C]` | Arithmetic mean of the same `sbp` pairs |
| `timestamps` | float64 `[T]` | Original source-bin right endpoints |
| `sequence_id` | integer `[T]` | Acquisition-block reset identity |
| `channel_ids` | integer `[C]` | Strictly increasing provider-local electrode IDs |

Pairs never cross a block or a time gap. Odd trailing source bins are discarded
and recorded. Export performs no statistical normalization. The feature file
contains no target, trial, go-mask, or scorer field. Channel IDs are local to a
person; they do not establish cross-person anatomical correspondence.

Targets remain the V2/V3 unit cursor-to-target direction at the second 10 ms
sample in each pair. A scored pair requires finite nonzero direction with norm
above `1e-12`, and both samples must lie in the same provider go interval.
The target sidefile contains only float32 `y[T,2]` and boolean `eval_mask[T]`.
This is a contemporaneous closed-loop intention proxy, not measured limb
velocity. Upstream voltage-filter and threshold-estimation causality remains
unverified; downstream prefix-only preprocessing cannot establish end-to-end
causality of the supplied features.

For `t3_blocks`, features cover the full causal history from session start
through the end of the last selected continuation block. Segment `start/end`
are global prepared-bin offsets; `source_start/source_end` retain original
10 ms indices, and `selected_for_evaluation` marks the fixed new blocks.
The fresh scorer file has zero target vectors and false masks everywhere else.
`known_prefix_target_path`, its SHA256, and `known_prefix_n_bins` separately bind
the existing development labels. The runner must slice those known labels to
its exact daily budget; the availability of a 600-second sidefile is not
permission to train on all of it.

## Metadata inventory before fresh outcomes

The existing 80 raw sessions contain 71 post-600-second tails totaling 28.7023
recorded hours. There are 380 complete blocks beginning after that prefix,
covering 26.4565 hours across 67 sessions and all nine people. Those larger tails
remain reserved except for the explicitly selected T3 blocks. Only block and
electrode metadata were opened for this inventory; target and feature values
from those tails were not examined.

The selected T3 continuation geometry is:

| Post-implant day | Selected blocks | Total selected seconds | Strictly earlier exposed sessions |
| --- | --- | --- | --- |
| 36 | 15, 16 | 596.60 | 0 |
| 37 | 6, 7 | 454.64 | 1 |
| 43 | 11, 12 | 229.50 | 2 |
| 44 | 14, 15 | 222.40 | 3 |
| 62 | 12 only | 298.40 | 4 |
| 79 | 11, 12 | 386.80 | 5 |
| 100 | 6, 7 | 596.44 | 6 |
| 406 | 9, 11 | 658.86 | 7 |

Day 62 has only one qualifying block; the shortage is retained. The selected
blocks total 3,443.64 seconds. These durations establish geometric availability,
not the amount of valid scorer data or attainable decoding accuracy.

The [existing exposed-prefix receipt](../results/v3-data-diagnosis.json) found 99.91656% zero TX4.5 channel-by-10-ms bins
in T3 day 36, while SBP was finite and positive. This motivates testing a lower
threshold without proving it will help. The provider's
[2025 preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC12236888/) reports persistently
low T3 yield and proposes wire-bundle tension with shallow electrode placement
as a possible explanation. That proposed mechanism is not established by this
software study. The provider's whole-block centering and whole-session scaling
are not suitable substitutes for prefix-only preprocessing in a causal replay.

## Commands and API

Run from the repository root with Python 3.12 and the existing environment.
Use fresh ignored output directories for reproduction.

```sh
export PYTHONPATH="$PWD/src"
python -m neural_experiments.v4.data prepare-development \
  --cache data/neural_v4/development
python -m neural_experiments.v4.data acquire \
  --cache data/neural_v4/new_sessions \
  --url-file data/neural_v2/transport-private.json
python -m neural_experiments.v4.data validate \
  --cache data/neural_v4/development --expected-role development
```

The URL file is optional and must contain only redirects observed through the
provider's ordinary authorized download flow. The local parallel orchestration
and exact byte ledger are preserved in ignored task evidence. The public CLI
uses sequential acquisition with the same fixed selection and aggregate bounds.

After the full study plan has been committed, pass its exact path and SHA256:

```sh
python -m neural_experiments.v4.data prepare-new_sessions \
  --cache data/neural_v4/new_sessions \
  --study-plan "$V4_STUDY_PLAN" --study-plan-sha256 "$V4_STUDY_PLAN_SHA256"
python -m neural_experiments.v4.data prepare-t3_blocks \
  --cache data/neural_v4/t3_blocks \
  --development-index data/neural_v4/development/index.json \
  --study-plan "$V4_STUDY_PLAN" --study-plan-sha256 "$V4_STUDY_PLAN_SHA256"
```

`load_index(path, expected_role=...)` accepts only `development`, `new_sessions`,
or `t3_blocks` and checks the role before referenced arrays. It validates the
exact parent-index identities, selected inventory, member and artifact hashes,
block partitions, and the separate known-prefix target binding for T3.
`load_features(path)` and `load_targets(path, n_bins)` remain separate APIs.
Synthetic ingestion tests cover synchronized pairs, target-free features,
continuation-label isolation, exact member ranks, and pre-open plan/role guards.
