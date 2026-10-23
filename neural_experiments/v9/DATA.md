# V9 data and target contract

V9 tests whether completed, direction-balanced calibration trials and a better-defined trial target can recover useful T2 decoding. Development consists of exactly 200 previously exposed prefixes: all 176 V8 development sessions and all 24 V8 assessment sessions. Eight participants contribute 24 sessions each; T3 contributes its existing eight. Every prefix remains capped at 30,000 paired 20 ms bins, or 600 recorded seconds. Development reuses the feature, target and behavior-context sidecars through authenticated immutable hard links. It does not reopen raw recordings or extend a prefix.

The CC0 source is [Dryad version 462702](https://datadryad.org/dataset/doi:10.5061/dryad.x0k6djj1h). Exact parent identities are checked before referenced artifacts are opened:

| Parent | SHA256 |
|---|---|
| V8 development, 176 sessions | `5c4b5ab0b0fa0ec28c2c5a1b746ca8e8ceee35b4797e2f7e2f66cc9d9dfdbf84` |
| V8 assessment, 24 sessions | `0ea2078c184e58014c9fc68d5fd5efa3855e93a12a82db6f2627174ae894c4b8` |

## What the targets mean

The preserved instantaneous target is the unit vector from cursor position to target position at the second 10 ms source sample of each complete 20 ms pair. Its mask requires finite nonzero displacement and both source samples inside the same provider go interval. This is task geometry, not measured limb velocity or an independently observed movement intention.

The provider reference implementation is pinned to commit [`93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834`](https://github.com/nptl-stanford/array-paper/tree/93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834). Its trial evaluation target is **the normalized mean of raw cursor-to-target displacement over a 300 ms window beginning at a selected reaction time**. Averaging unit directions would define a different target. The provider's onset-plus-40-ms reference supplies the distance used for far-trial eligibility. V8's fixed `reference40_direction` was a distinct secondary target proxy; it was not this window-averaged provider target.

The provider implementation checks that its evaluation window fits in the recording, but its window extractor does not enforce the supplied go-end bound. V9's protocol must require a window to remain inside one observed go interval and, for calibration, entirely inside purchased label time. This is a stricter validity condition than the provider extractor. Reaction-time choices, trial quotas, label budgets, and any target adaptation belong to the separately locked scientific protocol. Provider dSNR, shuffled trial cross-validation, and additional affine output calibration differ from this project's chronological held-out metrics. Whole-block centering and session-level normalization are not imported into prospective preprocessing.

Cursor and target positions have two columns, x then y. Their physical units and task scaling differ across participants. Screen-axis orientation, intended limb motion, trial success and assistance are not established by these fields. The provider's go intervals are identified from target changes and/or task state; the exported MATLAB records have no task-name, success or assistance columns. Neither cursor motion nor interval duration identifies those missing conditions.

## Exposure and behavioral context

Features contain TX3 and TX4.5 threshold-crossing counts summed across each pair, spike-band power averaged across that pair, timestamps, sequence IDs and channel IDs. Targets and behavior context remain separate files, accessible only through explicit loader opt-in. The context includes cursor and target positions, distance, provider go ordinal, exact onset time, age, paired bounds, right-censoring and onset/40-ms reference geometry. Feature-only loading never opens either label sidecar.

Exact onset time preserves the source 10 ms boundary on the accumulated recorded-time clock. Paired `trial_start_bin` and `trial_stop_bin` describe only complete 20 ms pairs. A trailing unpaired 10 ms sample can make the paired end earlier than the raw go end; completion checks must account for that difference. `trial_right_censored` identifies a go extending beyond the exposed segment. Recorded-bin time does not measure elapsed wall time across acquisition gaps.

The provider far-distance thresholds are 0.2 for T2, T3, T7, T9, T10 and T11; 160 for T5 and T6; and 5.6 for T8. T5 days 2107, 2126, 2133, 2149, 2595, 2609, 2637 and 2667 use its 0.4 override. Each index record preserves its applicable threshold.

Fresh conversion uses the existing V6 feature converter and V8 context converter. MATLAB decompression may mechanically materialize a whole compressed variable, but every numerical operation first copies the selected exposed source segment. Hashing a complete raw member authenticates bytes and does not inspect its signal values. No retained tail enters analysis. The independent auditor reconstructs context and the original target/mask from the exposed segments without calling the context producer.

## Fixed fresh cohort and acquisition

The [acquisition plan](acquisition-plan.json) fixes regular archive ranks **25–30 for T2, T7, T9 and T10**, requesting 24 sessions and six fresh T2 replicates. All 24 earlier regular members must match the exact V8 parents before a new member is retained. Archive order determines selection; it need not equal chronological order. Missing, short or failed cases keep their disposition, with no replacement or extension of the ranks. T3 continuations and other unexposed tails remain reserved.

Metadata feasibility was checked before acquisition using provider session identifiers, not decoding outcomes. The pinned provider `decoding_summary.csv` contains 42 T2, 35 T7, 117 T9 and 57 T10 unique sessions. Its Git blob is `b11566aa0cfe590f81f203a0d378d332e7c05970` and SHA256 is `2f1dbab5ec17b736322d38016127347410f1d942479c579d93c9bef90e696b62`. These counts support availability, not sufficient duration, trial coverage or accuracy. Actual ranks and shortages are established by archive streaming.

Limits are 32 GB of compressed response bytes across all attempts, 12 GB retained raw data and at most three workers. Synchronized reservations bound concurrent reads and raw writes; failed attempts retain their byte charges. Provider-issued redirects may be supplied in an ignored transport file. Partial streaming verifies prior and retained member hashes but does not verify the full provider archive digest.

Raw acquisition requires the committed acquisition lock. Fresh conversion additionally waits for the scientific plan, frozen producer and audited development choices to be committed. Fresh sessions never train one another. The cohort evaluates new recordings from known people, not untouched-person transfer. Public receipts contain hashes, counts, bounded acquisition costs and dispositions; samples, fitted weights, signed URLs and local absolute paths remain private.

```sh
PYTHONPATH=src:. .venv/bin/python -m neural_experiments.v9.data prepare-development \
  --cache data/neural_v9/development
PYTHONPATH=src:. .venv/bin/python -m neural_experiments.v9.acquire_parallel \
  --cache data/neural_v9/fresh --workers 3 \
  --url-file data/neural_v2/transport-private.json
```

These commands preserve the fixed cohort and refuse to overwrite an existing index. They do not authorize unlocking a scientific choice or replacing a failed case.
