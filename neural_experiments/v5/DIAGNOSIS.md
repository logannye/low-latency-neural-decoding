# Why V5 needs a cautious calibration policy

This diagnosis is **post-hoc**, informed by the already observed V4 failures. It
uses only the previously exposed first600 seconds from12 T2 sessions: nine V4
development recordings and three V4 assessment recordings. No raw recording,
untouched tail or T3 continuation was read. Existing V4 predictions reproduce
their saved scores to a maximum R² difference of3.33×10⁻¹⁶.

All new pilot comparisons below use the common **302–600 second** suffix. This
differs from V4's primary122–600 second endpoint. In particular, T2's daily120
mean is+0.0008 here, compared with+0.032 in the V4 primary report.

## Evidence and failed repairs

- Historical fitted mappings are unstable: the last three historical TX maps
  have pairwise coefficient cosines−0.008,−0.050 and−0.125. Combined-map cosines
  are0.210,0.006 and0.077. Correlated features and estimation noise can also
  change coefficients, so these numbers do not establish physical remapping.
- Latest-session-only history makes zero-label combined prior R² worse:
  −0.203 versus−0.104 with three histories. Increasing low-label adaptation
  strength also hurts: gamma1 at30s gives−0.390 versus gamma10 at−0.132.
- Calibration can learn the wrong direction bias. On day194, adding30s labels
  increases suffix MSE from0.4952 to0.5626: its mean-bias component rises from
  0.0245 to0.1052 while centered residual error improves slightly. Thirty-second
  prefixes contain only2–3 contiguous valid-go runs, not necessarily independent
  trials; two of the three sessions have mean target-direction norms near0.5.
- Dropping the original prior's SBP contribution reduces harm but does not
  establish useful decoding: mean R² changes from−0.104 to−0.035. Replacing
  target-prefix normalization with a shared historical normalizer is worse,
  yielding−0.997,−0.650 and−1.047 on the three later sessions.

The bounded pilot retained all1,728 fixed fits over two history counts, three
feature families, four budgets and six simple heads. It also retained288
prefix-selection decisions. A chronological last-third-prefix selector with a2s
purge performed poorly on the later three sessions:

| Labels | Fixed strongly regularized scratch | Prefix-selected model |
| --- | ---: | ---: |
|30s|−0.121|−0.253|
|60s|−0.008|−0.217|
|120s|+0.011|−0.035|
|240s|+0.006|−0.038|

Equal-octant or equal-go-run validation weighting did not recover useful mean
accuracy either. Thus a positive score in one short contiguous validation window
is insufficient evidence for stopping calibration. None of these post-hoc
results proves that T2 is undecodable or that any proposed V5 policy will recover
it. More calibration was helpful in older sessions but did not reliably repair
the later three.

## Consequence for the prospective study

V5 separates a30s unlabeled warmup from labels acquired afterward. Its six-model
foundation uses TX4.5/tau0.4 or TX4.5+SBP/tau0.2, each with a gamma10 historical
prior, a free-intercept ridge1 fresh fit, or a zero-intercept ridge10 fresh fit.
Adaptive candidates keep their normalizer fixed to the warmup. A policy must
explicitly permit collecting more labels or abstaining; thresholds are selected
using source-only chronological cases, then frozen before new assessment. The
policy is identical across participants. Its accuracy, label cost, acceptance
coverage and failures must be reported together.

The diagnosis pilots used prefixes starting at recording time zero; they are
**not evaluations of V5's subsequent-label schedule**. The V5 acceptance rules
and their fresh performance remain separate evidence.

## Original diagnostic provenance

Executed sources and aggregate receipts remain in the ignored
`work/v5-diagnosis/` directory, with all failures retained. They are private
diagnostic evidence, not a published replay package. The original files are
unchanged; the first run took9.47s and the followup3.45s with one BLAS thread.

| Artifact | SHA256 |
| --- | --- |
|t2_pilot.py|`f05d458dc92b89ddd10fb3efa21d84406f6f1c40e367dfb6e0b1d7ffeb385692`|
|t2_pilot.json|`979409e84ca0a0370b053877a3b9596e628aeebbc69079d271a0c7926ac537fb`|
|t2_followup.py|`174b8bf9cc579677bebae2a25436fd2015d8417644707a983aeb3978c9d6166a`|
|t2_followup.json|`027e77d48d00dfed68e7ed4eb196fffe7dc3e26877c0e59aafda3225752c33b4`|

The receipts include input feature/target hashes and frozen V4 source identities,
directly or through the parent receipt. No V4 code or original result was changed.
