# V3 data and assessment separation

V3 preserves the V2 intention-proxy endpoint and prepares two separate input
indexes. The development index contains the original 54 sessions, each capped
to its first 600 recorded seconds. Assessment selects regular decoding-session
members 7, 8, and 9 from each of the same nine pinned Dryad participant archives.
The [plan](plan.json) was committed as `9ab460f` before this acquisition, with
SHA256 `6d0f188bba2c9719bbb8dffece385d48a6b70e07b22835ee6413c89eee063392`.
Archive order is not chronological order. These are newly acquired sessions in
previously studied people, not a new participant cohort or uniformly future-day data.

The [data diagnosis receipt](../results/v3-data-diagnosis.json) records checks of
only previously exposed V2 windows and the primary provider code used to assess
their semantics. It found no target/index/channel mapping error in those checks.
The provider's trial-window dSNR procedure differs from V2's instantaneous
20 ms endpoint; the new assessment does not change that endpoint after seeing
results. SBP is an additional feature fixed in the V3 plan.

## Completed acquisition and validation

The development export and full identity/schema validation are complete:
54 sessions, 1,481,767 bins, index SHA256
`10bf61377fc43b1b6e77a4ae7238d80cde8894bb69af6f68388f54b8a5ac98dd`.
Its parent V2 index SHA256 is
`91de850304c4fb96096daa3f9ec19edab0372675c236a899005cf4308ec6c523`.

Assessment acquisition, preparation, and full identity/schema validation completed
on 2026-10-22. The [cohort receipt](../results/v3-cohort.json) records 26 sessions
across all nine participants, with exactly 30,000 bins per session: 780,000 bins
or 4.3333 recorded hours. The assessment index SHA256 is
`8850117ea21bfe939e869b53a49d5b8ccf7d40e905d51936d503759310f04c4d`.
The retained full raw sessions contain 13.3433 recorded hours; only their first
600 seconds per session were exported. The export contains 663,232 eligible
go-direction bins before model-specific prefix and evaluation-window restrictions.

T3's archive ended after eight regular session members, so rank 9 is unavailable.
Its ranks 7–8 remain selected; the other eight participants have all three
requested sessions. No replacement was made. All 26 fresh sessions have the
same electrode IDs, array labels, and grid coordinates as their participant's
V2 reference session. Selection and this metadata check used no model outcomes.
The prepared assessment inputs are now immutable for the locked evaluation.

## Feature and target contract

Each pure feature NPZ contains exactly:

| Field | Type and shape | Meaning |
| --- | --- | --- |
| `x_tx` | float32, `[T,C]` | Sum of two adjacent `tx_4_5` count bins |
| `x_sbp` | float32, `[T,C]` | Mean of the same two spike-band-power bins, in µV² |
| `timestamps` | float64, `[T]` | Source-bin right endpoints |
| `sequence_id` | integer, `[T]` | Provider acquisition-block identity for resets |
| `channel_ids` | integer, `[C]` | Increasing provider-local electrode order |

Both feature matrices use exactly the same within-block 10 ms pairs. Power is
averaged, while counts are summed. Odd trailing source bins are discarded and
accounted for; pairs never cross block boundaries. Export uses the first
`min(30,000, full prepared session length)` bins. Full raw source length and
block count remain in metadata. Upstream voltage filtering and threshold
estimation causality are unverified. Between-block wall-clock gaps are unknown.

The separate scorer/training target NPZ contains only float32 `y[T,2]` and
boolean `eval_mask[T]`. At the second source sample in each pair,
`d = target_position - cursor_position` and `y = d / ||d||` for finite `d` with
norm greater than `1e-12`; otherwise the stored target is zero. A scored pair
must have finite nonzero direction and both source samples in the same provider
go interval. Labels, go masks, trial identifiers, and target positions never
enter a feature file. This target is a contemporaneous intention proxy.

No statistical normalization is performed during export. The locked downstream
transform uses square-root counts and, for the combined family, `log1p` power;
normalization statistics come only from the exact allowed calibration prefix.
Full-block or whole-session centering would violate this evaluation design.
A session or budget with too little data is retained with an unavailable
result, not replaced because of duration or quality.

## Provenance and bounded acquisition

The source remains [Dryad DOI 10.5061/dryad.x0k6djj1h](https://datadryad.org/dataset/doi:10.5061/dryad.x0k6djj1h),
version 462702, CC0. Acquisition streams the fixed participant archives. It
hashes and verifies regular members 1–6 against the original V2 member identities
without storing duplicate files, then retains only ranks 7–9. A regular-session
rank is distinct from a tar-entry ordinal, which also counts directories and
other entries. Every retained MAT and prepared artifact receives a local SHA256.

The locked total compressed-transfer limit is 14,000,000,000 bytes, including
the preserved partial T5 prefix from switching the initial sequential downloader
to three synchronized workers. The retained-raw limit is 18,000,000,000 bytes.
Actual transfer was 9,836,893,277 bytes, including 758,710,272 bytes from the
interrupted initial T5 attempt. The complete selected raw members occupy
3,058,347,483 bytes. All nine participant acquisitions completed without a
failed final attempt. The orchestrator shares an exact response-byte counter
and reserves new member sizes before writes. It preserves its own source hash, importer hash, attempt
receipts, and accounting ledger. Whole-archive provider hashes remain metadata
unless explicitly verified; a consumed partial-prefix hash does not authenticate
unseen bytes. Signed provider redirects are private transport inputs and are
never published.

All original V2 files are unchanged. New raw files, prepared samples, target
sidefiles, signed transport URLs, and eventual model/prediction arrays remain
in ignored local caches. Public receipts contain only aggregate facts, source
identities, hashes, and dispositions.

## Commands and loading

Run from the repository root with Python 3.12 and the existing dependencies.
Choose fresh ignored output directories for a new reproduction.

```sh
export PYTHONPATH="$PWD/src"
python -m neural_experiments.v3.data prepare-development \
  --cache data/neural_v3/development --v2-index data/neural_v2/index.json
python -m neural_experiments.v3.data metadata --cache data/neural_v3/assessment
python -m neural_experiments.v3.data acquire \
  --cache data/neural_v3/assessment --v2-index data/neural_v2/index.json
python -m neural_experiments.v3.data prepare-assessment \
  --cache data/neural_v3/assessment --v2-index data/neural_v2/index.json
python -m neural_experiments.v3.data validate \
  --cache data/neural_v3/development --expected-role development
python -m neural_experiments.v3.data validate \
  --cache data/neural_v3/assessment --expected-role assessment
```

Ordinary Dryad browser verification may be required. In that case, pass
`--url-file` with an ignored participant-to-observed-redirect JSON map obtained
through the provider's normal download flow. Existing failure receipts are
preserved. The importer rejects a changed locked selection plan and never
silently replaces an unavailable participant or session.

`load_index(path, expected_role=...)` validates the role before referenced data
files, exact old/new session inventory, pinned provider mapping, acquisition
receipts, source members, feature/target hashes, and block partitions.
`load_features(path)` opens only the pure feature file; `load_targets(path)` is
for training and scoring. Prepared indexes are not overwritten in place.
