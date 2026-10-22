# V6 data selection and preparation

V6 evaluates fixed new archive sessions from eight previously studied people,
with T2 as the prespecified individual focus. No new T3 continuation domain is
included. The remaining T3 continuation blocks stay unexposed; T3 contributes
its eight original exposed prefixes to development only.

The [acquisition plan](acquisition-plan.json) was committed and pushed as
`92cf69a403919e49890975809bf1a9e90e464802` before acquisition. Its SHA256 is
`0849f2188f13b0df565809db1302ae673af938d4bf55e5f31b2c890960ac33c5`.
The provider remains Dryad DOI `10.5061/dryad.x0k6djj1h`, pinned version 462702,
under CC0-1.0. The [V5 data description](../v5/DATA.md) documents the unchanged
feature, target, channel and upstream-causality limits.

## Exposed development data

Development contains 128 distinct previously exposed session prefixes:
104 V5 development identities plus 24 V5 new-session identities. Each contains
only its first `min(600 seconds, duration)` of recorded observations. Eight
participants have 15 prefixes each; T3 has eight. No V4 or V5 continuation labels
are included. The exported total is 3,581,828 bins, or 19.89904 recorded hours.

| Parent | Index SHA256 |
| --- | --- |
| V5 development | `1fb2d9b3167e2c671e279dd89fe05ebb6f38633d4f3e6f3b56078a0c61621fb1` |
| V5 new sessions | `6ac718377da28dcb43ca127c8da6e6a78b40703ffc4b81358b06bb56ada4c5a4` |

V6 development index SHA256:
`357f8b917abec17f6354dac3cd93b4d87d24df0db4bf60afeee8bdbefbf84678`.
Prepared files are immutable hard links to their V5 parents. Their exact feature
and target hashes, lengths and block boundaries are preserved and checked by the
V6 loader. The new index binds the parent manifests and each reused artifact.

For any target recording, permitted history must have the same participant,
matching electrode layout and a strictly smaller numeric post-implant day.
Archive rank must never stand in for chronology. The development pool is fixed:
no fresh V6 assessment recording may enter another assessment case's history.

## Fresh-session acquisition

Selection is regular MAT archive ranks 16, 17 and 18 for T2, T5, T6, T7, T8, T9, T10 and T11,
independent of duration, neural quality or outcomes. Each stream authenticates
all 15 preceding regular members against the locked V5 provenance, then retains
only the selected new members. At plan lock, all eight previous receipts had
reached rank 15 without observing archive end; the subsequent member identities,
durations and availability were unknown. Missing ranks, short recordings,
transport failures and budget exhaustion receive explicit dispositions. None
permits replacement, an extended rank range or dropping a difficult participant.

The hard aggregate limits are 22,000,000,000 compressed response bytes and
12,000,000,000 retained raw bytes, with at most three concurrent workers. The
public [parallel downloader](acquire_parallel.py) reserves concurrent reads and
selected member storage under one shared lock. It retains attempted-transfer
accounting and failure receipts in the ignored cache. The public sequential
[importer](data.py) enforces the same locked aggregate ceilings.

Acquisition currently runs under the committed selection rule. Fresh outcome
preparation is blocked until the complete study plan is committed. A final
metadata-only receipt will record actual selected identities, transfer, artifact
hashes and all shortages after acquisition and preparation finish.

Streaming prefixes do not authenticate the provider's full-archive SHA256.
Receipts distinguish the local compressed-prefix digest, fully retained member
digests and the published but unverified full-archive digest. Optional signed
redirects must come from the provider's ordinary public download flow and remain
in a private URL mapping. They must never enter public manifests or reports.

## Export and loader contracts

Feature-only files contain `x_tx3`, `x_tx45`, `x_sbp`, `timestamps`, `sequence_id`
and `channel_ids`. The unchanged V5 converter sums adjacent 10 ms count pairs and
averages the same power pairs within each provider block. Pairing never crosses
a block or gap. The exporter learns no normalization. Fresh sessions are capped
at 600 recorded seconds; full raw length and block counts remain metadata.

Separate target files contain only `y` and `eval_mask`. Targets preserve the
normalized contemporaneous cursor-to-target direction and valid-go-pair rules.
This is an offline intention proxy, not observed limb velocity or a demonstration
of closed-loop control. Upstream voltage preprocessing causality remains unverified.

`load_index(path, expected_role=..., verify_hashes=True)` accepts only
`development` and `new_sessions`, validates role before referenced arrays, then
checks parent, member, receipt, artifact and segment identities. `load_features`
rejects target and mask fields. `load_targets` is a separate training/scoring API.
Fresh preparation requires the exact bytes of a committed full study plan and
its SHA256; acquisition authorization alone does not release outcome inspection.

From the repository root, using Python 3.12 and existing scientific wheels:

```sh
export PYTHONPATH="$PWD/src:$PWD"
python -m neural_experiments.v6.data prepare-development \
  --cache data/neural_v6/development
python -m neural_experiments.v6.data validate \
  --cache data/neural_v6/development --expected-role development
python -m neural_experiments.v6.acquire_parallel \
  --cache data/neural_v6/new_sessions --workers 3 \
  --url-file data/neural_v2/transport-private.json
```

Use fresh ignored output directories to reproduce prepared artifacts; existing
indexes and partial prepared files are never overwritten. The optional URL file
is only needed when the provider's normal flow supplies an observed redirect.
After the full study plan is committed:

```sh
python -m neural_experiments.v6.data prepare-new_sessions \
  --cache data/neural_v6/new_sessions \
  --study-plan "$V6_STUDY_PLAN" --study-plan-sha256 "$V6_STUDY_PLAN_SHA256"
```

Raw data, prepared samples, coefficients and sample-level traces remain ignored
local artifacts. Public evidence is restricted to code, metadata and aggregates.
