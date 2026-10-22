# V2 data: acquired cohort and preparation

As of 2026-10-22, all nine decoding participants have six retained complete
sessions, prepared and validated. The machine-readable [cohort receipt](../results/v2-cohort.json)
records the provider identity, selected member names and hashes, artifact hashes,
counts, acquisition limits, and calibration availability. The immutable private
index SHA256 is `91de850304c4fb96096daa3f9ec19edab0372675c236a899005cf4308ec6c523`.
The [transfer plan](transfer_plan.json) sets analysis windows; the complete
retained data are larger than those windows.

## Source and fixed selection

The source is [Dryad DOI 10.5061/dryad.x0k6djj1h](https://datadryad.org/dataset/doi:10.5061/dryad.x0k6djj1h),
pinned version **462702**, under CC0-1.0. That version's decoding release contains
nine people, fourteen arrays, and 729 sessions. This study uses **54 sessions**,
not the entire release. It includes every decoding participant: T2, T3, T5, T6,
T7, T8, T9, T10, and T11.

Selection takes the first six complete regular decoding-session MAT members in
each pinned participant archive, independent of decoding scores, recording
quality, or runtime. Selected sessions are then sorted by numeric post-implant
day, with session ID as a tie-breaker. Archive order is not chronological order;
for example, the earliest selected sessions are T3 day 36, T5 day 111, and T11
day 100. Short sessions remain included. This is a bounded deterministic cohort,
not a random sample of the 729 sessions.

| Participant | Selected post-implant days | Channels | Prepared bins | Scorable bins | Retained hours |
| --- | --- | ---: | ---: | ---: | ---: |
| T2 | 103, 110, 118, 124, 130, 131 | 96 | 796,977 | 682,901 | 4.4276 |
| T3 | 36, 37, 43, 44, 100, 406 | 96 | 690,273 | 623,009 | 3.8349 |
| T5 | 111, 113, 1015, 1036, 1048, 1167 | 192 | 359,799 | 332,187 | 1.9989 |
| T6 | 1001, 1006, 1011, 1013, 1018, 1020 | 96 | 80,189 | 62,456 | 0.4455 |
| T7 | 106, 135, 140, 162, 167, 205 | 192 | 587,413 | 566,234 | 3.2634 |
| T8 | 100, 106, 113, 130, 162, 1033 | 192 | 943,882 | 867,656 | 5.2438 |
| T9 | 105, 106, 121, 128, 132, 133 | 192 | 458,903 | 415,214 | 2.5495 |
| T10 | 104, 105, 109, 111, 112, 124 | 96 | 781,233 | 503,601 | 4.3402 |
| T11 | 100, 1001, 1017, 1022, 1025, 1057 | 192 | 327,723 | 301,994 | 1.8207 |

The retained data contain **5,026,392 bins at 20 ms**, of which
**4,355,252** pass the scorer mask, across **419 acquisition blocks**.
The retained duration is **27.9244 recorded hours**.
Durations count observations; the release does not supply wall-clock gaps
between blocks.

## Preparation and separation of features from targets

The importer uses `neural.tx_4_5` threshold-crossing counts for every session;
there is no feature fallback. It sums nonoverlapping pairs of 10 ms observations
within each provider block into a float32 count matrix `x[T, C]`. It never pairs
across a block boundary. All 10,052,823 source bins are accounted for:
39 odd trailing 10 ms bins were discarded, and
0 bins were outside the supplied blocks. The retained bins therefore satisfy
`2 * 5026392 + 39 = 10052823`.
No normalization or learned statistic is applied during preparation.

At the second 10 ms sample in each pair, let `d = target_position - cursor_position`.
The two-dimensional target is `y = d / ||d||` when `d` is finite and its norm is
greater than `1e-12`; otherwise the stored target is zero. The scorer mask also
requires both source observations to fall in the **same** provider go interval.
This is a contemporaneous intended-direction proxy, not measured limb velocity
or an independently measured intention signal.

Each feature NPZ contains only `x`, `timestamps`, `sequence_id`, and `channel_ids`.
Each separate target NPZ contains only `y` and `eval_mask`. Target and go-interval
information is used for training/scoring, never as decoder input. Sequence IDs
mark acquisition blocks and require state resets. Timestamps are source-bin
right endpoints; their construction does not establish wall-clock continuity
between blocks. Channels retain provider-local order. Equal numeric channel IDs
across people do not establish neuron identity or anatomical correspondence.

The strict index loader verifies the provider mapping, archive-order selection,
member identities, feature and target artifact hashes, and chronological block
partitions. The feature and target loaders separately validate array geometry,
dtypes, masks, channel order, and within-block timestamp spacing. Validation
also confirmed identical channel IDs across each person's six selected sessions.
Upstream voltage filtering and threshold-estimation causality remain unverified;
causal claims concern processing from the supplied count features onward.

## Retained data versus analysis windows

The locked plan uses at most **600 recorded seconds per session** for source
training/validation and target evaluation. Across all 54 selected sessions,
those capped inputs contain **1,481,767 bins** or
**8.2320 hours**, counted once rather than multiplied by
methods, seeds, budgets, or leave-one-person-out folds. These input windows
include observations used only as causal history or calibration context.

The primary future-day comparison uses the five later selected sessions per
person: **45 sessions**, **1,243,374 capped bins**,
**6.9076 recorded hours**, and **1,080,377 scored bins**.
The secondary within-session comparison uses the common interval from 300
seconds to `min(600 seconds, earliest-session end)`. Across the cohort that is
**107,170 bins**, **0.5954 recorded hours**,
and **94,886 scored bins**. Eight people have at least 50 scored
bins in this suffix. These are window-accounting counts, not performance results.

Calibration uses the earliest selected session without substitution, at budgets
of 30, 120, or 300 recorded seconds. A budget needs a complete prefix and at
least 50 training rows after the common 62-bin exclusion at each block start.
Future-day eligibility does **not** require a same-session suffix. T6's earliest
session lasts **224.46 seconds**: its **30- and 120-second** future-day comparisons
remain eligible, its **300-second** budget is unavailable, and all its secondary
same-session-suffix metrics are unavailable. T5's earliest session lasts 343.4
seconds, leaving 43.4 seconds and 1,971 scored bins in its common suffix. Neither
participant is replaced or removed because of these limitations.

## Acquisition provenance and storage

The nine published decoding archives total **79,428,705,009 bytes**. Acquisition
streamed gzip/tar prefixes and stopped after each sixth complete session member.
It consumed **6,026,756,096 compressed bytes** for the final selection, plus a
separately preserved **118,095,872-byte** initial T3 feasibility probe. The probe's
first member was reused rather than stored twice. The complete task cache after
preparation was about 5.72 GiB, below the configured 20 GiB limit.

Every retained complete MAT member and every prepared artifact has a local
SHA256. Consumed compressed prefixes also have local hashes. The provider's
published checksum covers each **full archive**, so all nine full-archive
verification flags are **false** for this partial acquisition. Member and prefix
hashes establish the bytes retained; they do not verify unseen archive bytes.
The public receipt labels these limits explicitly.

Initial blocked download attempts and later successful attempts are preserved
separately. Successful acquisition used redirects observed through Dryad's
ordinary public browser download flow. Expiring signed URL queries remain in an
ignored private transport file and are not part of the public receipt. No
account credentials, verification bypass, or full-archive mirror was used.

## Reproduction and privacy

Use Python 3.12 with the installed dependencies described in the [reproduction
guide](README.md). The importer commands are:

```sh
export PYTHONPATH="$PWD/src"
python -m neural_experiments.v2.data metadata --cache data/neural_v2
python -m neural_experiments.v2.data acquire --cache data/neural_v2 --sessions 6 --max-cache-gib 20
python -m neural_experiments.v2.data prepare --cache data/neural_v2 --max-cache-gib 20
python -m neural_experiments.v2.data validate --cache data/neural_v2
```

For a fresh reproduction, choose a fresh ignored cache directory and use it
consistently in all four commands. Do not rewrite a cache that an active run is
using. If ordinary Dryad browser verification is needed, acquisition supports a
private participant-to-observed-redirect JSON map via `--url-file`; alternatively,
`--archive-directory` accepts local complete provider archives and verifies their
full checksums. Default acquisition bounds each compressed stream to 4 GiB.

Raw MAT files, prepared arrays, transport URLs, model weights, and prediction
traces remain under ignored local `data/` or `results/` directories. The public
cohort receipt contains only provider metadata, counts, selection/accounting
facts, relative paths, and hashes. No sample arrays are included. The study is
an offline transfer experiment using an intention proxy; it establishes neither
live-control performance nor clinical benefit or broad population generalization.
