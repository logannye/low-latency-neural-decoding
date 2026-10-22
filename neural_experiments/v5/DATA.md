# V5 data selection and preparation

V5 keeps two distinct domains: new archive sessions in eight known participants,
and optional new continuation blocks in previously studied T3 sessions. Neither
is an untouched-participant replication. Selection depends on archive order and
block metadata, never duration, signal quality, or decoding outcomes.

The [acquisition plan](acquisition-plan.json) was committed at `db5ca7b` before
V5 downloads. Its SHA256 is
`43a55022369f44d76659d453e20880009692bfa80591aa49d0b54677cb8c1f47`.
The provider remains Dryad DOI `10.5061/dryad.x0k6djj1h`, pinned version 462702,
under CC0-1.0. The [V4 data description](../v4/DATA.md) records the unchanged
feature, target, channel and upstream-causality limitations.

## Exposed development data

Development contains 104 distinct previously exposed session prefixes, each
limited to its first `min(600 seconds, duration)` of recorded observations:
80 V4 development sessions plus 24 V4 new sessions. T3 contributes its eight
original prefixes only; none of its V4 continuation outcomes enters this index.
The other eight participants each contribute 12 prefixes.

The V5 development index contains 2,909,665 bins and has SHA256
`1fb2d9b3167e2c671e279dd89fe05ebb6f38633d4f3e6f3b56078a0c61621fb1`.
Prepared files are immutable hard links to existing V4 files, preserving their
exact feature and target hashes without duplicating sample storage. V5 keeps its
own role manifest and provenance. Its loader also compares development artifact
hashes, counts and segments with the pinned V4 parents.

Pinned parents:

| Parent | Index SHA256 |
| --- | --- |
| V4 development | `aa6bf303c746edde0fb4d0a3a937a0fb07afb849270f2c79093fc021a9c2c500` |
| V4 new sessions | `76293562fd6e586e7d6f5506c41ebda97241c971e18984c22489b69ac29558cc` |
| V4 T3 continuation | `7f8908c54312814ce41b7e743042e85fb890a443ee267f8fed937ae66b27de69` |

## New-session acquisition

The fixed new-session selection is regular MAT members 13, 14 and 15 for T2,
T5, T6, T7, T8, T9, T10 and T11: 24 requested identities. Streaming authenticates
all 12 preceding members against retained source hashes before accepting new
members. Only selected new members are saved. Archive order is not day order;
all downstream history checks must use strictly smaller numeric post-implant
days and matching electrode layouts. Missing ranks, short recordings and other
unavailable cases receive explicit dispositions without replacement.

The aggregate limits are 20,000,000,000 compressed response bytes and
12,000,000,000 retained raw bytes. Concurrent workers share exact transport
accounting and raw-member reservations. Failed attempts count against the same
transfer budget and remain in private receipts. Published full-archive hashes
are not claimed verified when acquisition stops after a prefix. Observed signed
redirects remain private; public provenance uses provider IDs and hashes.

Acquisition completed on 2026-10-22 with all 24 requested ranks, no failed
participants, no missing ranks and no substitutions. Actual transfer was
13,332,971,520 compressed bytes; retained raw storage was 2,265,009,299 bytes.
The [public cohort receipt](../results/v5-cohort.json) binds acquisition accounting,
member and prepared-artifact hashes, parent indexes, selected geometry and every
disposition. All 135 exported session entries match their participant-specific
electrode ID, array ID, x and y reference layout.

The new-session index passed full identity and schema validation. Its SHA256 is
`6ac718377da28dcb43ca127c8da6e6a78b40703ffc4b81358b06bb56ada4c5a4`;
it contains 672,163 bins, or 3.73424 recorded feature hours. The fixed `[272,600)`
window geometry totals 6,921.90 seconds before go masks and decoder burn-in.
The minimum exported duration is 13,268 bins (265.36 seconds), so no selected
session is below the 30-second input requirement. T6 day1048 is below the fixed
272-second evaluation start and remains unavailable. T6 day1057 has only 4.20
seconds of window geometry; whether it meets the fixed minimum valid-pair count
is an assessment question, not a reason to replace it.

T8 day38, selected at archive rank13, has no strictly earlier same-participant
development session. Its empty history pool is retained under the predefined
fallback. These are newly accessed sessions, not necessarily later acquisition
days. For example, T11 day122 is rank15, following days1139 and1181 in archive
order; historical training still follows numeric day.

## Separate T3 continuation

For every cached T3 session, the fixed rule selects the next two whole blocks
immediately after its last V4-selected block. The metadata inventory is:

| Post-implant day | V5 selected blocks | Selected seconds | Shortage |
| --- | --- | ---: | --- |
| 36 | 17 | 298.24 | One block unavailable |
| 37 | 8, 9 | 575.30 | None |
| 43 | 13, 14 | 220.38 | None |
| 44 | 16, 17 | 266.00 | None |
| 62 | None | 0 | Two blocks unavailable |
| 79 | 13, 14 | 357.40 | None |
| 100 | 8, 9 | 596.46 | None |
| 406 | 12, 13 | 596.52 | None |

Thus 13 requested blocks are available across seven sessions, totaling 2,910.30
recorded seconds. The exhausted day62 remains an explicit session disposition.
These are metadata geometry counts, not valid-label counts or accuracy evidence.
The prepared T3 index passed identity, schema and label-isolation validation.
It contains 567,959 feature-history bins and 145,515 selected-block bins; its
SHA256 is `e368324e7a8b5216e781bfe68d75a94170762417c46982bfde7a0bb6df4c45c9`.
Features preserve history from session start through the final selected block.
Fresh scorer targets and masks are zero/false outside the selected new blocks.
Known-prefix labels are separately bound to the development artifact; the runner
must enforce its exact permitted prefix budget.

## Export and loader contracts

Feature files contain only `x_tx3`, `x_tx45`, `x_sbp`, `timestamps`, `sequence_id`
and `channel_ids`. Counts sum within-block adjacent 10 ms observations; power
averages the same pair. Pairing never crosses a block or gap. Odd trailing source
bins are accounted for. No statistical normalization is learned during export.
Scorer files contain only `y` and `eval_mask`, preserving the unit cursor-to-target
direction and within-go valid-pair semantics. These remain offline intention
proxies; upstream voltage-filter causality is unverified.

`load_index(path, expected_role=..., verify_hashes=True)` accepts `development`,
`new_sessions` or `t3_blocks`, checks the role before opening referenced arrays,
and authenticates parent, member, receipt, artifact and block identities.
`load_features(path)` rejects target/mask fields. `load_targets(path, n_bins)` is
a separate training/scoring interface.

Fresh export requires a full study-plan path and SHA256 whose exact bytes are
committed in this repository. The [full study plan](plan.json) was committed at
`4b39a60`, with SHA256
`87c417f915259e7f77b7999217d3a2461884fffd175c676188e20c346472390e`,
before fresh preparation. Importer and execution sources were frozen at `c41f793`. Acquisition alone does not authorize fresh sample
inspection or scoring. Source selections and their independent audit must be
published before the one-shot assessment. Raw files, prepared arrays, fitted
models and prediction traces remain ignored local artifacts.

Run from the repository root using Python 3.12 and existing scientific wheels:

```sh
export PYTHONPATH="$PWD/src"
python -m neural_experiments.v5.data prepare-development \
  --cache data/neural_v5/development
python -m neural_experiments.v5.data acquire \
  --cache data/neural_v5/new_sessions \
  --url-file data/neural_v2/transport-private.json
python -m neural_experiments.v5.data validate \
  --cache data/neural_v5/development --expected-role development
```

Use new ignored output directories when reproducing an existing export; indexes
are never overwritten. The optional URL file may contain only redirects observed
through the provider's ordinary download flow. The private parallel orchestrator
and its source snapshot supplement the public sequential CLI with shared bounds.
After the full study plan is committed:

```sh
python -m neural_experiments.v5.data prepare-new_sessions \
  --cache data/neural_v5/new_sessions \
  --study-plan "$V5_STUDY_PLAN" --study-plan-sha256 "$V5_STUDY_PLAN_SHA256"
python -m neural_experiments.v5.data prepare-t3_blocks \
  --cache data/neural_v5/t3_blocks \
  --development-index data/neural_v5/development/index.json \
  --study-plan "$V5_STUDY_PLAN" --study-plan-sha256 "$V5_STUDY_PLAN_SHA256"
```
