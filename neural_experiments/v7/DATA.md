# V7 exposure and information boundaries

The recovery experiment reuses exactly 152 session prefixes exposed through V6:
128 V6 development sessions and 24 V6 assessment sessions. The latter are now
development data; none is a fresh V7 test. There are 18 prefixes for each of T2,
T5, T6, T7, T8, T9, T10 and T11, and eight for T3. All selected cases are retained.

The source is the existing public CC0 Dryad release `10.5061/dryad.x0k6djj1h`,
pinned version 462702. No new download, raw MAT parsing or continuation extraction
is required for this stage. Its 4,264,975 prepared bins contain at most 600
recorded seconds per session. Feature arrays and target arrays remain separate.

| Artifact | SHA256 |
| --- | --- |
| V6 development parent index | `357f8b917abec17f6354dac3cd93b4d87d24df0db4bf60afeee8bdbefbf84678` |
| V6 new-session parent index | `dfe5ed95bf7f43b0bb51f3a11d163505e99641a4affc86daf38591fbdc5af534` |
| V7 exposed union index | `4b2c0c8b47a2caf1a2135b28dcea197a26c8f6006c19f243ff2fed23c5e1362e` |
| Recovery plan | `28ead298081c41b6028bffa0cf9aa5ee0eeb7811917400863aff350525aa7671` |

Preparation uses immutable hard links and checks the exact parent metadata and
sample-file hashes. It replaces each inherited provenance map with a reference
to its immediate V6 parent. Both immediate and earlier lineage remain available
through the bound parent indexes. The independent verifier checks those maps
separately from unchanged record fields.

## Predefined primary cases

The greatest six numeric T2 acquisition days are **214, 228, 236, 237, 242 and
250**. Each has a complete 600-second exposed prefix. These identities were
selected by chronology before the new model outcomes, not by measured recovery.
Archive order is never treated as chronology.

Nine T6 prefixes end before the first 272-second forecast: days 1001, 1011,
1013, 1018, 1020, 1034, 1039, 1048 and 1088. They retain all method and window
dispositions. No fit is executed for those recordings, so their diagnostic label
usage is zero. This accounting records labels actually used by fits; it is not
a prospective calibration schedule or an interruption-time measurement.

## Causality and target meaning

The unchanged prepared target is unit cursor-to-target direction on valid go
pairs. It is an offline intention proxy. The neural features are threshold
counts and spike-band power in 20 ms bins, transformed and filtered causally
with resets at acquisition-block boundaries. Upstream feature extraction
causality and acquisition latency remain unverified, as in V6.

The runner loads the exposed session locally, but fitting receives only the
target prefix ending two seconds before its next forecast. The first 30 seconds
or the explicitly declared past training interval supplies normalization. No
later observations enter fitted statistics. The information restriction is an
API/numerical boundary, not filesystem isolation from targets.

Labels from an earlier scored window may train a later update after the fixed
delay. Trial identifiers are not present in the prepared export; neither the
embargo nor window separation is claimed to make trials independent. Statistics
resample sessions or participants, never individual neural bins.

The original protocol reserved raw tails and later ranks unless the gate passed.
That gate failed. The separate [V7B amendment](CONTINUAL_PROTOCOL.md) explicitly
changes the continuation rule after exposed results, with an acquisition and
policy plan committed before new outcomes. It opens only fixed archive ranks
19–21 in the eight non-T3 participants. Raw continuation tails and the remaining
T3 data stay reserved. Sample traces, model coefficients, complete local indexes
and source snapshots remain ignored. Public outputs contain aggregate metrics
and artifact hashes.

## V7B fresh acquisition

The fresh acquisition authenticates all 18 prior archive members for each of
T2, T5, T6, T7, T8, T9, T10 and T11 against the exact V6 parents. Only regular
ranks 19–21 are retained, regardless of duration or task difficulty. Archive
rank fixes selection; numeric post-implant day determines reporting order.
Preparation retains at most the first 600 recorded seconds, separates features
from targets, and applies the unchanged target/feature conversion. Whole-person
threshold exclusion concerns source selection; fresh sessions are still from
known participants and do not establish transfer to untouched people.

The acquisition plan bounds total compressed transfer at 28 GB, retained raw
files at 12 GB and concurrency at three workers. The initial public-API attempt
was blocked by the provider and transferred zero archive bytes. An existing,
still-valid provider-issued public redirect map from the earlier study resumed
acquisition through the same frozen downloader. Its digest matches the earlier
transport lineage; signed URLs remain private. All attempts and budget charges
are preserved. No browser challenge was circumvented.

The continual plan was committed before preparation; audited source choices
are published before assessment. The importer and execution source are checked
against their frozen hashes. Fresh scoring includes every selected recording,
with unavailable scores explicitly retained for insufficient observations.
