# Prospective adaptive calibration

The [machine-readable plan](plan.json) was committed as `4b39a60`, before fresh
outcome preparation. SHA256:
`87c417f915259e7f77b7999217d3a2461884fffd175c676188e20c346472390e`.
The acquisition commitment predates it at `db5ca7b`; the initial execution source
was committed at `c41f793`. Source policy choices must be independently verified
and published before assessment. Later failures cannot change this experiment's
thresholds, candidates, masks, or success criteria.

## Question and first-principles design

History is useful when an earlier neural-to-direction relationship remains
predictive. A decoder that fits its own historical labels well does not establish
that relationship will persist. V5 therefore tests historical priors on strictly
later *historical* pseudoqueries before allowing a zero-fresh-label decision.
Each reliability probe is itself out of session; the current query's targets
never contribute to its reuse decision.

Collecting more samples also does not necessarily add useful calibration
information. Adjacent 20 ms bins are correlated, and a short run can cover only
a few directions. V5 separately checks directional coverage and prediction on
a later, purged portion of the requested calibration interval. It compares
validation error with both a zero prediction and the calibration-training mean,
so merely learning a prevalent direction is insufficient for acceptance.
Two-second group agreement is a conservative heuristic, **not an independence
assumption or a confidence bound**. Fresh evaluation tests whether it works.

The [T2 diagnosis](DIAGNOSIS.md) documents failed alternatives. Recent-history-only
fits, weaker regularization, shared historical normalization, and simple
prefix-validation winner selection did not reliably repair the exposed failures.
Coefficient instability suggests poor mapping transfer, but correlated features
and estimation noise prevent treating it as proof of physical neural remapping.

## Information flow and decisions

1. Observe 30 seconds of neural features without requesting labels. Compute the
   fixed query normalizer. This warmup is required even for zero-label reuse.
2. For each of two historical prior families, inspect up to three strictly
   earlier chronological probes. Reuse the lowest-error eligible family when
   at least two probes pass the selected mean-error ceiling and worst-error cap.
3. Otherwise request labels prospectively for 30, 60, 120, then at most 240
   seconds after warmup. Decision times are 60, 90, 150, and 270 seconds.
4. At each stage, train candidates on the first half of the requested interval;
   validate after a two-second purge, ending at the current decision time.
   Require sufficient rows, directional coverage, retained validation groups,
   improvement over both nulls, and agreement across at least two thirds of
   those groups. Refit the passing candidate on the entire requested interval.
5. If none passes at the cap, emit a zero-output **abstention**. If a checkpoint
   is unavailable, abstain at the recording's end and retain both requested and
   available label cost. An abstention does not count as recovered decoding.

The six candidate heads combine TX4.5 counts alone or counts plus spike-band
power with prior-centered ridge, ordinary daily ridge, or strongly regularized
zero-intercept daily ridge. The four policies vary only historical mean-error
ceiling (0.40 or 0.45) and prefix-validation improvement margin (0.025 or 0.05),
measured as coordinate-mean squared error. All transformations are causal and
reset at acquisition boundaries. Query normalization is fixed before labels;
historical normalization may use the entire earlier recording.

The policy accepts a history-evidence object and a lazy prefix callback. It
rejects extra outcome fields. The callback supplies only the requested label
interval, and the runner makes the fresh decision before loading scorer targets
or evaluating counterfactual controls. Tests poison unavailable labels and
exercise stopping, chronology, shortages, coverage, and publication boundaries.

## Source selection and fresh evaluation

Development uses 104 previously exposed session prefixes, each capped at 600
observed seconds. For each target person, policy ranking excludes all outcomes
from that person. Among policies within 0.03 mean R² of the daily120 control,
choose the one with least requested label cost. If none qualifies, maximize the
predeclared accuracy-minus-cost objective. Accuracy is averaged within person,
then equally across source people; cost includes short and unscored cases.

The new-session domain contains fixed archive ranks 13–15 for eight known
participants. Archive order is not acquisition-day order. All history selection
uses strictly earlier numeric days and matching electrode layouts. The fixed
history pool never adds another fresh assessment session.

Every method and stopping time uses the **same 272–600 second score window**.
Controls are zero output, fixed combined-feature history with zero or 30 fresh
labels, and daily calibration with 120 or 240 labels. Daily controls receive all
unlabeled normalization observations available at their own stopping time, making
them stronger comparators than using a 30-second normalizer for every method.
The target remains an offline cursor-to-target direction proxy.

T3 contributes a separate negative-control domain: the next two previously
unused whole blocks after V4's fixed continuation selection. Seven sessions
have remaining blocks, one does not; every requested disposition is retained.
These are new observations from known sessions. No intermediate or previously
scored continuation labels enter calibration or policy reliability.

Primary accuracy includes abstentions as zero predictions. Coverage and negative
accepted-case counts are reported separately, preventing selective reporting from
making refusal look like recovery. The uncertainty calculation resamples eight
participant means; T2 and T3 session intervals are descriptive within-person
summaries. The three fresh T2 sessions cannot establish broad population recovery.

## Predeclared gates

- Useful adaptive decoding: mean R² at least 0.10, bootstrap lower bound above
  zero, at least six of eight participant means positive, gain over zero at
  least 0.05, and mean angular error at most 80 degrees.
- Reduced calibration: the useful gate passes, the lower paired R² interval
  against daily120 exceeds −0.05, and mean requested label cost is at most
  90 seconds. This is an engineering criterion, not a powered clinical trial.
- T2 recovery: all three requested sessions scored, mean R² at least 0.05,
  at least two accepted positive sessions, gain over zero at least 0.05, and
  a positive lower paired-gain interval over fixed zero-label history.

Inference receives only neural features. Numeric exports are reloaded before
replay. Independent references reconstruct fitting, history probes, prefix
validation, decisions, source exclusion, final scores, costs, and bootstrap gates.
Raw data, labels, coefficients, model files and sample traces remain local.

## Relation to prior work

Historical diversity and decoder robustness have precedent in
[Sussillo et al., 2016](https://www.nature.com/articles/ncomms13749).
[Latent-dynamics alignment](https://www.nature.com/articles/s41467-025-59652-y)
and [unsupervised cursor-decoder recalibration](https://www.nature.com/articles/s41551-025-01536-z)
address related instability problems with different models and evaluation
settings. V5 implements a small, auditable label-request controller; these
papers do not validate its particular thresholds or its offline results.
