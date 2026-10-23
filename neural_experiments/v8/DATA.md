# V8 data and target contract

V8 asks whether T2's persistent decoding failure is explained by its cursor-to-target label, limited task-direction coverage, or the neural representation. Development uses exactly the 176 session prefixes already exposed through V7: 152 V7 development records plus all 24 V7 assessment records. They are now development data. Eight participants contribute 21 sessions each; T3 contributes its existing eight sessions. Every prefix remains capped at 30,000 paired 20 ms bins, or 600 recorded seconds. No retained recording tail enters development.

The public source is the CC0 Dryad release [10.5061/dryad.x0k6djj1h](https://doi.org/10.5061/dryad.x0k6djj1h), pinned version 462702. Parent indexes are authenticated before any referenced arrays are opened:

| Parent | SHA256 |
|---|---|
| V7 development, 152 sessions | `4b2c0c8b47a2caf1a2135b28dcea197a26c8f6006c19f243ff2fed23c5e1362e` |
| V7 fresh assessment, 24 sessions | `34f51825b1f1b4b7f8144fcfe93822f61295a2a013f0cd564fff610228d20e47` |

## Enriched development export

The neural features and existing targets are reused through immutable hard links, with exact parent identities retained. Feature files contain only TX3, TX4.5, spike-band power, timestamps, sequence IDs and channel IDs. Target files contain unit cursor-to-target direction and its valid-go mask. New behavior context lives in a third sidecar, never returned by feature-only loading. Labels and context require explicit opt-in through `load_case`.

The provider stores compressed MATLAB variables. Reading a selected variable can mechanically decompress its complete container. Before numerical analysis or export, the converter copies only the authenticated exposed source segments. Decompression does not authorize analysis of the remaining samples. Raw hashing authenticates byte identity; the audit never computes signals, targets or statistics from retained tails.

The context preserves cursor and target position at the second source sample of each existing 20 ms pair, distance, original provider go-interval ordinal, go age, exact onset time, paired go bounds and right-censoring. A pair belongs to a go interval only if both of its 10 ms source samples belong to the same interval. The independent audit reconstructs these fields directly from selected raw segments, then verifies the original target and valid-go mask byte-for-byte.

`trial_onset_seconds` expresses the exact 10 ms provider onset on the accumulated recorded-bin clock. It avoids rounding an onset at 29.99 seconds into the purchased interval starting at 30 seconds. `trial_start_bin` and `trial_stop_bin` instead describe the complete paired rows available inside that interval. Source-relative feature timestamps preserve acquisition gaps; recorded-bin time does not establish elapsed wall time across those gaps.

Two reference directions are explicit. `onset_direction` uses cursor-to-target geometry at the raw go onset. `reference40_direction` uses onset plus 40 ms, matching the provider's reference convention; its availability flag is false if the required sample is outside the exposed go interval. This reference is not causal at onset. The study may use it only in the declared later neural phase and purchased-label scope. Neither reference changes the primary instantaneous target.

The provider's go intervals are identified from target changes and/or task state, but the exported records do not provide task names, trial success, assistance level or measured intended limb velocity. Cursor movement can reflect the existing decoder and assistance. It is not an independent ground truth for intention. No task-state or success labels are inferred here.

## Provider reference and distance units

The cached provider implementation is pinned to commit `93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834`. Its decoding reference uses distance at onset plus 40 ms, reaction-time selection, trial-based cross-validation and additional calibration. Its reported metric is therefore different from this project's temporally held-out instantaneous target metric. Its whole-block centering and session-level normalization are not imported into causal candidate preprocessing.

The provider's far-distance thresholds are participant-specific because position units differ. They are 0.2 for T2, T3, T7, T9, T10 and T11; 160 for T5 and T6; and 5.6 for T8. T5 days 2107, 2126, 2133, 2149, 2595, 2609, 2637 and 2667 use the provider's 0.4 override. Each record stores the exact applicable threshold. A common threshold across participants would mix incompatible coordinate units. The original primary target retains its existing nonzero-distance rule; far-distance subsets are separately named evaluations.

## Fresh cohort and access order

The [acquisition plan](acquisition-plan.json) fixes regular archive ranks 22–24 for T2, T5, T6, T7, T8, T9, T10 and T11. It authenticates all 21 preceding regular members against the exact V7 parents before retaining new members. Archive order fixes selection; it is not necessarily chronological. Missing, short and failed cases retain explicit dispositions without replacement. T3 continuations and all retained recording tails remain reserved.

The locked limits are 32 GB of compressed response bytes across all attempts, 12 GB of retained raw files and three workers. Concurrent reads and raw writes reserve budget before I/O. Failed attempts retain their byte charges. Public provider-issued redirects may be supplied in an ignored private transport file; signed URLs are never public artifacts. Partial archive streaming verifies prior and retained member hashes, but cannot verify the provider's full-archive digest. These evidence types remain distinct.

Raw acquisition begins only after the acquisition plan is committed. Fresh export additionally requires the committed scientific plan and frozen producer. Audited source choices must be published before fresh scoring. Fresh sessions never train one another. The fresh cohort tests unseen sessions from known people; it does not establish transfer to an untouched participant population.

Local reproduction uses the existing environment:

```sh
PYTHONPATH=src:. .venv/bin/python -m neural_experiments.v8.data prepare-development \
  --cache data/neural_v8/development
PYTHONPATH=src:. .venv/bin/python -m neural_experiments.v8.audit_data \
  --index data/neural_v8/development/index.json \
  --output neural_experiments/results/v8-development-data-audit.json
PYTHONPATH=src:. .venv/bin/python -m neural_experiments.v8.acquire_parallel \
  --cache data/neural_v8/fresh --workers 3 \
  --url-file data/neural_v2/transport-private.json
```

The first command refuses to replace an existing index. The acquisition command enforces the committed acquisition-plan identity. These commands do not authorize changing the locked cohort or study choices. Sample arrays, context traces, raw files and fitted coefficients remain ignored; public data evidence contains hashes, counts and acquisition dispositions only.
