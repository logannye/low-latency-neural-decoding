# Frozen assessment data release

The assessment gateway is implemented and tested with synthetic fixtures. This
implementation does not itself authorize opening sealed recordings. Before a real
export, the coordinator must record the freeze's exact SHA-256 and explicitly
release that artifact. A digest proves byte identity, not authority, model quality,
or truthful training provenance; the training run and its independent audit supply
the latter evidence.

`validate_freeze` and `select_assessment_assets` inspect metadata, source, configuration
and model artifacts only. `export_assessment` is the separate operation that opens
NWB arrays. It creates a fresh local directory under
`data/neural_h1/prepared_assessment/<stage>-<freeze SHA prefix>/`. The existing raw,
prepared-development, upstream and metadata symlinks remain read only. An existing
assessment export is never overwritten; a failed export retains a failure record
and cannot pass the completed-index loader.

## Two frozen stages

| Stage | Model training days | Evaluation days | Provider role |
|---|---|---|---|
| `early` | S0-S4 | S5 | `held-in-calib` |
| `later` | S0-S5 | S6-S12 | `held-out-calib` |

The later freeze references the early freeze by path and independently recorded
SHA-256. Its recipe, primary metrics, data identities and decoder-source hashes
must equal the early freeze. Model identities change only through the declared
fixed-recipe refit including S5. This permits S5 to become refit data after the early
assessment; it does not permit redesign in response to its score. S6-S12 labels
remain scorer-only for the primary assessment. An optional supervised-prefix study
uses its own explicit label access and separate results.

The machine-readable freeze has these fields:

```json
{
  "schema_version": 1,
  "status": "frozen",
  "stage": "early",
  "created_at": "2026-10-22T00:00:00+00:00",
  "source": {
    "git_revision": "<full committed revision>",
    "files_sha256": {"src/neural_decoder/example.py": "<SHA-256>"}
  },
  "recipe": {"path": "<repository-relative recipe>", "sha256": "<SHA-256>"},
  "models": [{
    "name": "<unique model name>",
    "kind": "<linear or tcn>",
    "path": "<repository-relative artifact>",
    "sha256": "<SHA-256>",
    "training_days": ["S0", "S1", "S2", "S3", "S4"],
    "regime": "frozen_zero_calibration"
  }],
  "data": {
    "upstream_pin": "770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f",
    "manifest_sha256": "<raw-cache manifest SHA-256>",
    "schema_audit_sha256": "<raw-cache schema audit SHA-256>",
    "split_manifest_sha256": "<raw-cache split manifest SHA-256>"
  },
  "evaluation_days": ["S5"],
  "provider_split": "held-in-calib",
  "primary_metrics": {
    "decoding": "mean_of_day_variance_weighted_r2",
    "runtime": "p99_complete_step_service_ms"
  }
}
```

This is a schema illustration, not a usable freeze. The source map must include
every Python file under `src/neural_decoder/`, as well as the frozen recipe when
created by the assessment runner. Each source hash must match both its bytes at
the recorded Git commit and its current working-tree bytes. The current HEAD may
advance for documentation-only commits; implementation/configuration changes
invalidate the freeze. Artifact paths must remain inside the current repository.
For the later stage add `parent_freeze: {path, sha256}` and use the corresponding
later roles and newly refitted models. JSON duplicate keys and nonfinite constants
are rejected.

Before reading any sample arrays, the gateway checks the pinned provider UUIDs,
source hashes, day mapping, provider split, channel/output schema and exact stream
inventory. Early assessment consists of two S5 streams; later assessment consists
of fourteen streams, two per day. Minival is never an assessment input. The day
mapping is extracted from the hash-verified pinned upstream source; relabeling
an asset in an index cannot change its role.

Exported feature files contain only counts, timestamps, sequence IDs and channel
IDs. Labels, masks and trial changes are in separate scorer files. Both keep the
pinned official loader's bin/timestamp semantics. All released arrays are compared
with the pinned loader before a completed index is issued. Count values use uint32;
an actual count exceeding the upstream uint8 representation would fail parity and
require an explicit compatibility decision rather than silently wrap.

`load_assessment_index(index_path, freeze_path, expected_sha256, root=..., cache=...)`
validates the release, receipt, complete inventory, role bindings and every artifact
hash before any array loader runs. `load_assessment_files` then returns the same
`(index, [(record, counts, targets, mask), ...])` layout as the development loader,
for an authorized scorer or fixed-recipe refit. Predictors still receive features
only. The original development loader continues to reject sealed roles.

## Optional supervised-prefix feasibility

Metadata-only inspection confirms that all fourteen later streams support the
prespecified 0/5/15/30-second per-file budgets. Their lengths range from 46.06 to
54.26 nominal seconds. Every budget arm scores the identical suffix beginning at
bin 1500, the 30-second boundary, leaving 16.06 to 24.26 seconds per file. The
supplied evaluation mask will further determine scored samples; that count is not
inferred from duration.

| Prefix budget per file | Prefix bins per file | Nominal seconds per day, two files |
|---|---:|---:|
| 0 seconds | 0 | 0 |
| 5 seconds | 250 | 10 |
| 15 seconds | 750 | 30 |
| 30 seconds | 1500 | 60 |

The calibration job receives only its own preceding label prefix. Ordinary state
processes every preceding neural bin identically for all arms, including the gap
between a shorter label prefix and the common suffix; state resets at each file.
A later file's labels cannot calibrate an earlier prediction. Report nominal
prefix duration and actual available/scored label bins separately. These budgets
are chronological recording time, not a promise that every bin supplies a scored
training label. The primary zero-new-day-label assessment finishes separately
before this optional supervised analysis.

Synthetic tests cover immutable freeze checks, committed/current-source identity,
documentation-only commits, source/config/model changes, source-role relabeling,
metadata-only validation, parent freezes, common suffixes, separated feature files,
repeat-export rejection, corrupt indices and retained failures. No real sealed
array is needed by the tests.
