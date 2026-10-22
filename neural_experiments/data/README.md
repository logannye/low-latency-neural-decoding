# FALCON H1 data identity and schema audit

This directory contains metadata and hashes for the local H1 study. Neural recordings,
behavior arrays, prepared examples and prediction traces stay in the ignored cache.
The dataset is one participant, so none of this evidence establishes cross-participant
transfer. The local assessment described by the protocol is not an official FALCON score.

## Pin, terms, and storage

The live DANDI inventory was checked on October 22, 2026 before downloading. Dandiset
[000954](https://dandiarchive.org/dandiset/000954/draft) had only a mutable `draft`, with
40 assets totaling **102,258,934 bytes**. `h1_manifest.json` pins every asset UUID, path,
provider digest, byte size, retrieval timestamp, and independently calculated SHA-256.
All 40 local SHA-256 values equal the provider's `dandi:sha2-256` value. The DANDI draft
status was `Invalid` for contributor-name formatting; its returned asset validation
error list was empty. This is metadata status, not a statement about data quality.

The license field is `spdx:CC-BY-4.0`. The dataset description also states this Data Use
Agreement, which the local research workflow honors:

Downloading constitutes agreement to the provider terms. They forbid attempts
to identify or contact participants, restrict contact about individual results,
prohibit further disclosure or redistribution, and require team members and anyone
receiving access to comply. The exact provider text is retained with the local
acquisition metadata; the linked DANDI page is the authoritative terms source.

No raw/prepared data are redistributed. The complete provider metadata and receipts
are retained in the ignored cache. Dataset citation: Ye, Joel; Jennifer L. Collinger;
Robert Gaunt (2024), *FALCON Benchmark H1: Human 7DoF Reach and Grasp Motor BCI*, DANDI
000954, draft. The data pin here is the immutable list of asset identities, not a claim
that the mutable draft URL is immutable.

The upstream source is pinned to
[`snel-repo/falcon-challenge@770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f`](https://github.com/snel-repo/falcon-challenge/tree/770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f).
`upstream_manifest.json` records individual file hashes, including the retained optional
`requirements.txt` 404. Source uses the MIT license reproduced in `FALCON_LICENSE.txt`.
The local binning implementation adapts that source with attribution. Only the needed
source files were cached; the complete benchmark package/dependencies were not installed.

At ingestion, raw files used 102,258,934 bytes; compressed prepared arrays/index used
7,257,949 bytes; the source, metadata and library-schema caches added under 0.4 MiB.
This is well below the 5 GiB study budget. There is one raw cache and one prepared cache.

## Actual schema and timing

All 40 NWB files have **176 rows in `units`, in ID order 0 through 175**, and target shape
`(T,7)`. These IDs identify rows; an electrode/channel anatomical mapping is not supplied
by that table and is not inferred. The metadata for both kinematic series gives output
order **`tx,ty,tz,rx,g1,g2,g3`** and unit **`arbitrary`**. The dataset-level description
mentions translation, roll and grasp dimensions but does not supply an unambiguous
`g1/g2/g3` anatomical mapping. Reports use the seven source labels.

Each file has `OpenLoopKinematics`, `OpenLoopKinematicsVelocity`, `TrialNum`, and
`eval_mask`. The loader uses velocity targets and evaluates bins where the mask casts
to `True`, despite the mask description's ambiguous wording about ignoring timesteps.
All files have a stored NWB `starting_time=0`, data `offset=0`, and `rate=0.02`. In
conventional NWB usage `rate` is samples per second; these H1 files store a 20 ms interval
there. The pinned official loader explicitly constructs
`offset + arange(T) * rate`, and our exporter preserves that exact rule. It does not
reinterpret the value as 0.02 Hz or replace the loader's offset with starting_time.
No file has explicit kinematic timestamps. Consequently the loader-derived timeline
is uniform at 20 ms; this does not establish that the original acquisition had no gaps
before conversion.

A timestamp is the end of a nominal bin. Histogram intervals are left closed/right
open, except that NumPy includes the last right edge in the final interval. The first
bin covers `[-0.02,0)` on the supplied timeline. A gap longer than 20 ms discards its
earlier portion and retains only the proximal 20 ms interval. A short interval uses
its actual width. No fabricated missing bins are replayed. Tests cover those endpoints,
gaps, short intervals, duplicate timestamps and a 300-spike overflow fixture.

For **all 11 S0-S4 calibration files**, exported counts, targets, trial-change flags
and masks exactly match the unchanged functions extracted from the pinned official
loader. Using AST extraction avoids importing Hydra/global configuration; the loader
functions themselves are not rewritten. The data and mapping source SHA-256 values
are checked before extraction. The official count output is uint8; the exporter uses
uint32 to prevent silent wraparound. Observed development counts range from 0 to 8,
so the wider dtype does not change these files' count values. Count ranges on sealed
days are deliberately not inspected yet.

State must persist across trials and reset at file boundaries. The raw trial numbers
and derived trial-change flags are exported only in the targets/scoring file; the
feature-only file contains exactly `counts`, `timestamps`, `sequence_id`, `channel_ids`.
The predictor does not receive trial labels, behavior labels, or masks.

## Day roles, dates and overlap

`split_manifest.json` defines forward folds S0-S1→S2, S0-S2→S3, S0-S3→S4. All crops and
augmentations inherit their source day. The prepared index has 11 calibration files
from S0-S4, 112,488 bins, and 109,603 scored bins. S5-S12 are metadata-only and sealed;
no neural or behavioral value arrays from those days were read or exported for modeling.
Shape/dtype, IDs, dates, durations, mask/trial shape and spike-storage length are schema
metadata, not an evaluation. There is no final score or early/later assessment yet.

| Day | Calibration files | Bins | Nominal seconds | Provider date | Role |
|---|---:|---:|---:|---|---|
| S0 | 2 | 17,275 | 345.50 | 1925-01-01 | Development |
| S1 | 3 | 25,524 | 510.48 | 1925-01-08 | Development |
| S2 | 2 | 23,378 | 467.56 | 1925-01-13 | Development |
| S3 | 2 | 22,338 | 446.76 | 1925-01-15 | Development |
| S4 | 2 | 23,973 | 479.46 | 1925-01-19 | Development |
| S5 | 2 | 23,503 | 470.06 | 1925-01-20 | Sealed early |
| S6 | 2 | 4,784 | 95.68 | 1925-01-26 | Sealed later |
| S7 | 2 | 5,101 | 102.02 | 1925-01-27 | Sealed later |
| S8 | 2 | 4,816 | 96.32 | 1925-01-29 | Sealed later |
| S9 | 2 | 4,902 | 98.04 | 1925-02-02 | Sealed later |
| S10 | 2 | 4,672 | 93.44 | 1925-02-03 | Sealed later |
| S11 | 2 | 4,912 | 98.24 | 1925-02-06 | Sealed later |
| S12 | 2 | 5,303 | 106.06 | 1925-02-09 | Sealed later |

These are de-identified provider timestamps, not an assertion that recordings actually
occurred in 1925. Their date labels span **39 days**, with early days spanning 19 days
and later days spanning 14 days. No stronger calendar interpretation is inferred.
The 14 later calibration files are separate streams; two files per day do not become
one continuous stream. Chronological order uses numeric day IDs and set identity;
canonical official scorer assembly uses lexicographic day keys separately.

There are also 13 minival files, one for every early calibration file. The provider
explicitly calls minival a subset of calibration. For 11 development pairs, velocity,
mask and trial-number arrays exactly equal the calibration prefix `[0,minival_T)`.
`minival_overlap.json` records exact offsets and UUID pairs. S5's two pairs share session
metadata and provider subset status, but their value arrays remain uninspected. Minival
is excluded from training/selection and independent evidence throughout.

## Reproduction and information boundaries

From the repository root with the neural optional dependencies installed:

```sh
python -m neural_decoder.data acquire --cache data/neural_h1 --manifest neural_experiments/data/h1_manifest.json
python -m neural_decoder.data prepare --cache data/neural_h1
NEURAL_H1_INDEX=data/neural_h1/prepared/index.json python -m pytest -q tests/neural_decoder/test_h1_data.py tests/neural_decoder/test_splits.py
```

The pinned acquisition command requests exact UUIDs and verifies the recorded provider
and local digests. It fails if they differ; it never substitutes a newer draft. If a
cache already has `manifest.json`, acquisition automatically reuses that pin. On a new
unpinned cache, `acquire` checks current metadata/license/inventory before downloading.
`prepare` audits every file but only materializes S0-S4 calibration streams, compares
them to the pinned loader, and emits metadata audit/split/overlap evidence. Rebuilding
the prepared cache is an explicit operation; experiments bind to its recorded hashes.
The pynwb schema cache is kept under the same ignored study directory.

`prepared/index.json` is for the training/scoring orchestrator. It names separate
`features/<file_id>.npz` and `targets/<file_id>.npz` files with independent hashes.
`load_index` verifies the source-manifest hash and each asset's pinned day/file/provider
identity. Relabeling a sealed UUID as S0 cannot bypass the guard. Feature loaders reject
unexpected keys, corrupt sequence IDs/channel maps and changed hashes. Training checks
its allowed days again. Target/mask arrays are not part of a predictor's input contract.
The split guard rejects minival, duplicate streams, file/day mismatches, unauthorized
days, and attempts to authorize sealed days through the development path.

The tests passed locally with installed NumPy, h5py and pynwb; no Rust/native rebuild
was performed. The public download sandbox restriction and an initial pynwb default
cache permission failure were resolved by reviewed public-data access and a local
schema-cache path. Compact failure records remain under the ignored data cache.
