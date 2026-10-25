# Aggregate evidence index

These files contain aggregate scores, resource/latency summaries, dataset and
artifact identities, and independent audit receipts. Raw recordings, prepared
sample arrays, model weights, sample-level predictions, fitted per-file calibration
offsets, raw timing distributions and the real-data replay remain in the ignored
local cache. The public code can recreate and audit those artifacts after authorized
data acquisition. A hash identifies retained bytes; it is not independent access
to the underlying data or a scientific validation by itself.

| Experiment | Results | Independent check |
|---|---|---|
| Original ridge / TCN development | [Development report](../REPORT.md) | [Ridge](ridge-audit.json), [TCN](tcn-audit.json) |
| Bounded mean correction | [Aggregate](adaptation-development.json) | [Portable arithmetic/scoring audit](adaptation-audit-v2.json) |
| Training channel dropout | [Aggregate](augmentation-development.json) | [Nine-model audit](augmentation-audit.json) |
| Frozen S5 assessment | [Scores](final-early.json), [freeze](early-freeze.json) | [Ten-trace audit](early-assessment-audit.json) |
| Frozen S6–S12 assessment | [Scores](final-later.json), [freeze](later-freeze.json) | [Seventy-trace audit](later-assessment-calibration-audit.json) |
| Separate prefix calibration | [Aggregate, offsets removed](calibration.json) | [224-trace audit](later-assessment-calibration-audit.json) |
| Same-weight runtime | [All 15 blocks](runtime.json) | [Timing/parity audit](runtime-audit.json) |
| Dynamic-int8 conversion | [Nine-model quality](compression-quality.json), [eight runtime blocks](compression-runtime.json) | [Quality](compression-quality-audit.json), [runtime](compression-runtime-audit.json) |
| Exact final H1 deployment | [Report](../H1_FINAL_CLOSURE_REPORT.md), [all-seed quality](h1-final-closure-v1/quality.json), [50 runtime blocks](h1-final-closure-v1/runtime.json), [two 90,000-bin soaks](h1-final-closure-v1/soak.json) | [Quality](h1-final-closure-v1/quality-audit.json), [runtime](h1-final-closure-v1/runtime-audit.json), [soaks](h1-final-closure-v1/soak-audit.json), [lifecycle](h1-final-closure-lifecycle-audit-v1.json) |
| Sparse M2 transfer | [Aggregate](m2-transfer.json) | [Independent preparation, refit and scoring](m2-audit.json) |

The final H1 deployment study has separate [source/artifact bindings](h1-final-closure-v1/bindings.json), an [export manifest](h1-final-closure-v1/manifest.json), and [aggregate tables](h1-final-closure-v1/tables.md). Its quality and trace audits passed; its overall provisional 2 ms service-p99 goal failed, and the NumPy/int8 soaks retained 1/5 missed deadlines respectively. Local validation passed 1,361 tests. Code and aggregate evidence are published in [draft PR #11](https://github.com/logannye/low-latency-neural-decoding/pull/11), whose [check list](https://github.com/logannye/low-latency-neural-decoding/pull/11/checks) tracks hosted software validation by revision. Private H1 recordings and measurements remain local.

[run-provenance.json](run-provenance.json) records the original run-manifest hashes,
source snapshots and diffs, configuration identities, dependency versions and
allowlisted runtime environment. Development runs may include recorded source
diffs. The primary runtime used committed revision `8d7220e`; the early freeze
used the same revision and the later freeze used `4f30a3d`, with identical frozen
implementation/recipe hashes. Compression runtime used `8290ed6`; the M2 runner
and configuration were committed at `d80e4d0`. Documentation and independent-audit
work continued alongside later runs; exact implementation identities remain
bound separately. New reproduction runs need new model/configuration hashes and
new freeze receipts, rather than reusing the published artifact identities.

The [original adaptation receipt](adaptation-audit.json) is retained alongside the
portable auditor's independently recomputed v2 receipt. Every reported audit passed. Public receipts are byte-identical to the retained
receipts; the calibration aggregate deliberately omits fitted offsets. The original
release's hosted validation is available in the repository's GitHub Actions and
pull request history. Linux CI checks packaging and synthetic contracts; the
M4 Max measurements above are local performance observations.

The M2 quality failure, failed bounded-adaptation keep rule, weak H1 output, and
one ONNX deadline miss are retained in the [final report](../FINAL_REPORT.md).
