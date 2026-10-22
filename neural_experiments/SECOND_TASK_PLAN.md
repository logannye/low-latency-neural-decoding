# M2 secondary-task plan

Status: the following design was fixed before array access, committed as
`d80e4d0`, and executed after the required H1 assessment and quiet timing window.
The seven pinned assets were acquired and the independent audit passed. Quality
transfer failed with this sparse training; see [M2_REVIEW.md](M2_REVIEW.md).
The run snapshot preserves this document's exact pre-execution version. No
decoder source files were changed.

## Bounded design and claim

Use all seven complete public M2 minival NWBs: fit the already selected causal
ridge recipe on 2020-10-19 Run1+Run2, then evaluate once on 2020-10-20,
2020-10-27 and 2020-10-28. The first two future dates contain two files each; the
last contains one. Run a training-target mean comparator. Keep history 1,
regularization 0.1, the 12-tap exponential FIR with tau 12 bins, training-only
normalization, and file resets. No tuning, adaptation, neural training or
reuse of H1 weights is planned.

This tests transport of the implementation and fixed fitting recipe to another
task, plus local forward-day quality from very sparse training. It does not test
cross-participant generalization of a trained model, official hidden-test
performance, or long-term robustness. The animal is one male rhesus macaque,
MonkeyN, with M1 recordings and two finger-group movement targets. Provider
split names do not denote separate animals.
[Dataset metadata](https://api.dandiarchive.org/api/dandisets/000953/versions/draft/info/).

Minival overlaps the corresponding calibration split, but that causes no
cross-role overlap here: only minival is used, each asset once, with strictly
ordered training/evaluation dates. The pinned converter takes the first **two
nonblank trials per recording**. Thus training has four trials, and future-day
evaluation has 4/4/2 trials. This is a small numerical transfer demonstration;
negative quality is retained without expanding data or searching settings.
Metadata does not give actual recording durations. Measure duration, sample
count, scored count and trial count after authorized ingestion, before fitting.
[Converter](https://github.com/snel-repo/falcon-challenge/blob/770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f/preproc/m2_preproc.py),
[constants](https://github.com/snel-repo/falcon-challenge/blob/770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f/preproc/nwb_create_utils.py).

## Exact acquisition inventory

All paths share
`sub-MonkeyN-held-in-minival/sub-MonkeyN-held-in-minival_ses-`
and suffix `_behavior+ecephys.nwb`. Links below are metadata, not array reads.
The adjacent [locked configuration](configs/m2-minival-transfer-v1.json)
contains complete paths, download URLs, provider asset metadata, SHA-256 and
DANDI ETag for each asset.

| Role | Date and run | Bytes | Asset metadata |
|---|---|---:|---|
| Train | 2020-10-19 Run1 | 21,196,248 | [9df908a2](https://api.dandiarchive.org/api/assets/9df908a2-3381-4486-b52d-358f68424033/) |
| Train | 2020-10-19 Run2 | 15,855,264 | [08b354ca](https://api.dandiarchive.org/api/assets/08b354ca-d8e1-4c67-a84c-893e7c082454/) |
| Evaluate | 2020-10-20 Run1 | 14,449,040 | [5f29fc64](https://api.dandiarchive.org/api/assets/5f29fc64-32f8-41b1-93e7-82043fd87f72/) |
| Evaluate | 2020-10-20 Run2 | 14,278,472 | [5704e8e3](https://api.dandiarchive.org/api/assets/5704e8e3-a8e9-42e1-9c94-f413cea74ba0/) |
| Evaluate | 2020-10-27 Run1 | 17,341,136 | [bc2760ef](https://api.dandiarchive.org/api/assets/bc2760ef-f23e-486c-9159-4264aeb514d0/) |
| Evaluate | 2020-10-27 Run2 | 17,360,784 | [c32a6c5a](https://api.dandiarchive.org/api/assets/c32a6c5a-db80-47ce-9501-8afaa1a2e62e/) |
| Evaluate | 2020-10-28 Run1 | 23,694,592 | [3381f481](https://api.dandiarchive.org/api/assets/3381f481-440a-4d80-92e1-0edd9927e060/) |

Raw total: **124,175,536 bytes (118.423 MiB)**. Training accounts for
37,051,512 bytes. Cap added storage at **384 MiB** including raw files, prepared
arrays, predictions, models and provenance; reuse the existing interpreter and
upstream source cache. One ridge fit and one constant fit; one thread; no new
grid. Check actual total study storage against the existing 5 GiB provisional
budget before acquisition. Whole M2 is 15,916,372,200 bytes in 20 assets; its
embedded `assetsSummary` is stale. Only a mutable draft exists, currently marked
invalid due to metadata validation; UUIDs, digests and exact metadata must be
pinned instead of relying on the draft name.
[Live inventory](https://api.dandiarchive.org/api/dandisets/000953/versions/draft/assets/?page_size=100),
[versions](https://api.dandiarchive.org/api/dandisets/000953/versions/).

Dataset metadata declares open access and **CC-BY-4.0**. Preserve provider
attribution and URLs. No additional data-use restriction was found in the
retrieved M2 description; do not import H1's distinct agreement into M2. FALCON
code is MIT licensed and its existing license/source cache is reused.
[Dataset terms](https://api.dandiarchive.org/api/dandisets/000953/versions/draft/info/),
[code license](https://github.com/snel-repo/falcon-challenge/blob/770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f/LICENSE).

## Loader and numerical acceptance

The pinned config expects 96 neural inputs and 2 outputs at 20 ms/bin. The
converter creates unit IDs 0–95 for threshold crossings, with output names
`index` and `mrs` (middle/ring/small grouped fingers). It computes a numerical
gradient of smoothed normalized position without a time divisor and stores
unit `AU`; do not relabel those values as physical velocity per second.
Actual NWB row order, names, units and timestamp alignment remain ingestion
checks. [Config](https://github.com/snel-repo/falcon-challenge/blob/770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f/falcon_challenge/config.py),
[converter](https://github.com/snel-repo/falcon-challenge/blob/770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f/preproc/m2_preproc.py).

The converter describes target timestamps as bin right edges, while the
official M2 loader passes `is_timestamp_bin_start=True`. Preserve the loader's
counting convention, including endpoint behavior, and set each observation's
release time to target timestamp +20 ms. This supports causality relative to
released observations, not zero acquisition delay relative to the target
timestamp. Confirm independent spike binning agrees exactly with the pinned
loader; fail visibly on mismatched cadence or overflow. Retain all bins,
continue state through trials and unscored bins, reset only per file, and give
the predictor neural counts alone.
[Loader](https://github.com/snel-repo/falcon-challenge/blob/770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f/falcon_challenge/dataloaders.py).

Report variance-weighted R2 after concatenating scored rows within each real
calendar day, all per-output values/counts, the equal-day mean and population
standard deviation. Independently check sklearn and the pinned FALCON helper.
Compare full-file numerical convolution with incremental predictions; exercise
future perturbation, mask/label isolation, chunk partition and file reset.
Retain every prediction, model, source snapshot, input digest and any failure.
Do not remap dates into fake H1 S labels. No speed claim follows from this
numerical check; any later timing must use the primary study's explicit
measurement boundary and run after other workloads finish.

## Hidden evaluation readiness

The public EvalAI challenge and phase pages responded on 2026-10-22. They list
Minival and Test phases through 2099; the displayed Test limit is 6/day,
50/month and 100 total. This confirms public configuration only, not worker
health or account eligibility.
[Evaluation](https://eval.ai/web/challenges/challenge-page/2319/evaluation),
[phases](https://eval.ai/web/challenges/challenge-page/2319/phases).

The pinned README requires a Docker image, team participation and EvalAI token.
A presence-only local check found Docker CLI on PATH, no EvalAI CLI on PATH,
and no standard `~/.evalai/token.json` file. No credential file was opened,
no authentication was attempted and no account, team or submission was created.
Hidden evaluation is therefore **not ready** from this checkout; authenticated
access and a container smoke test remain separate work.
[Submission instructions](https://github.com/snel-repo/falcon-challenge/blob/770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f/README.md),
[official token-path definition](https://github.com/Cloud-CV/EvalAI-CLI/blob/master/evalai/utils/config.py).

## Execution and validation

The standalone runner is `neural_experiments/transfer_m2.py`. It rejects altered
roles/settings, saves a fresh result directory, validates source/input hashes,
and records a failed manifest on an interrupted or unsuccessful run. With the
orchestration release and source commit in place, run:

`PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python neural_experiments/transfer_m2.py --released-after-primary-work`

Five synthetic tests passed before acquisition, covering exact bin endpoints,
overflow/cadence rejection, forbidden scoring arguments, future causality,
chunk/reset parity, unequal-length day aggregation and a two-trial NWB fixture.
An initial collection attempt exposed PyNWB's automatic home-cache creation;
the runner and fixture now import it lazily with a local cache. The corrected
version passed all five tests and Ruff before acquisition. The scientific run
completed on its first attempt, followed by independent artifact reconstruction.

