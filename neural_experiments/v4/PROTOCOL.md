# V4 evaluation protocol

The exact [machine-readable plan](plan.json) was committed as `b714f12` before
fresh outcome preparation. Its SHA256 is
`f00419baf098ab5e42d6757714bd43cc5107579edf2a7c915c24a4f6ff7d8aba`.
The acquisition plan preceded it at `a8dfce0`; its threshold field spelling was
corrected to the provider's actual `tx_3` field at `2f1e943`, before extraction.

## Why history might reduce fresh labels

A 30-second prefix can contain thousands of bins but only a few sustained
directions. Adjacent bins are strongly correlated. Fitting a new unconstrained
mapping to that prefix can therefore estimate its directional bias more reliably
than its generalizable neural mapping. Earlier recordings provide additional
directional coverage, but their electrode statistics differ from the query day.

V4 normalizes each historical recording in its own coordinates and combines
their sufficient statistics with equal recording weight. For eligible historical
days \(s\), with augmented normalized design \(Z_s\), labels \(Y_s\), and labeled
row count \(n_s\):

\[
G_0=\frac1S\sum_s Z_s^T Z_s/n_s,\qquad
B_0=\frac1S\sum_s Z_s^T Y_s/n_s,
\]
\[
W_0=(G_0+\operatorname{diag}(1,\ldots,1,0))^{-1}B_0.
\]

Only the last three strictly earlier days of the same person and channel layout
are eligible. Selection occurs before checking label counts. An insufficient day
is omitted with a disposition, without substituting an older day. No eligible
history gives a zero prior. Historical fits use fixed observed-bin indices
divisible by five after applying the common burn and go mask; the representation
still processes every observation. This bounds redundant correlated matrix work.

The current session uses its first 30 seconds of unlabeled features for fixed
normalization. For a fresh labeled prefix of 0, 30 or 120 seconds:

\[
W=(G+\Lambda)^{-1}(B+\Lambda W_0),\qquad
\Lambda=\gamma\operatorname{diag}(1,\ldots,1,0.1).
\]

Fewer than 50 eligible fresh rows returns the prior unchanged. The matched
scratch control uses the same representation, normalization, penalty and fresh
labels, with \(W_0=0\). Historical supervised labels are additional information;
the zero-fresh-label arm is not a model trained without labels.

## Source selection

All 80 development recordings were previously exposed through their first
`min(600 seconds, duration)`. Exploratory T3 and history pilots informed this
finite design. Those pilot results are development evidence.

The 16 global configurations cross TX4.5/TX3 counts, with/without spike-band
power, a past-only five-second baseline correction or none, and adaptation
penalty 1 or 10. Count-only smoothing is 400 ms; combined smoothing is 200 ms.
For each target person, rank 30-second chronological pseudo-query scores using
only the other eight people, averaging sessions within person before people.
Use the resulting recipe unchanged at all three fresh-label budgets.

The 16 targeted T3 daily configurations cross TX4.5/TX3, baseline correction or
none, ridge penalty 1/10, and free or exactly constrained zero intercept. They
fit 120 seconds and validate on `[302,600)` of source recordings. Each T3 query
selects using strictly earlier T3 source sessions; the earliest query falls back
to equal-person non-T3 source scores. Its own session and later T3 sessions never
rank its targeted recipe. Apply that same recipe at 30 seconds as secondary.

All source scores, exported fits, replay predictions and selections must pass an
independent audit and be published before fresh assessment. The producer's code,
plan and development-index hashes are checked again at the assessment boundary.

## Fresh assessment and declared gates

| Domain | Fixed observations | Primary question |
| --- | --- | --- |
| Eight known people | Archive members 10–12; score valid pairs from 122 s through min(600 s, duration) | Does 30 s adaptation give useful accuracy, and retain 120 s daily accuracy? |
| T3, eight known sessions | First two whole blocks beginning at or after 600 observed seconds, or available subset | Does the targeted 120 s daily repair recover useful T3 accuracy? |

All arms within a domain use identical masks. T3 feature state replays from
recording start; only the initial 30/120-second known prefix can train the query
model. Fresh scorer labels exist only in the selected later blocks. Archive
position is not acquisition chronology. No fresh assessment recording becomes
another assessment recording's history.

The new-session useful-accuracy gate requires all eight people, mean R² ≥0.1,
participant-bootstrap lower bound >0, at least six positive person means,
mean gain ≥0.05 over both zero and prefix-mean controls, and angular error ≤80°.
The zero-fresh-label arm uses the same gate and discloses its 30-second unlabeled
warm-up. A 75% reduction in fresh labeled duration retains accuracy only if the
30-second arm also passes its useful gate and the paired lower confidence bound
against daily120 exceeds −0.05. This is a descriptive engineering criterion,
not a powered clinical noninferiority trial.

The T3 recovery gate requires all eight sessions, mean R² ≥0.05, session-bootstrap
lower bound >0, at least six positive sessions, gain ≥0.05 over zero, paired gain
over daily120 with lower bound >0, and angular error ≤80°. Resampling sessions
from one person does not produce participant-level evidence. Both domains use
10,000 bootstrap draws with seed 20261023; no pooled headline joins the domains.

## What the experiment does not establish

The label is a contemporaneous cursor-to-target direction proxy in existing
closed-loop recordings. Offline prediction accuracy does not establish online
task performance, intended limb velocity or clinical benefit. The participants
are already known to development, and T3 continuation uses known session
identities. This is a locked retrospective assessment on newly accessed outcomes,
not prospective or untouched-participant confirmation. Upstream neural feature
extraction causality and acquisition latency remain unverified. CPU processing
time is reported separately from filter response.
