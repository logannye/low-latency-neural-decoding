# V7: local recoverability before continual recalibration

This study first asks whether fresh, recent labels can restore useful T2 decoding.
It is a conditional research plan: a sparse continual controller and fresh-session
assessment proceed only if the exposed-data recovery gate passes. A failed gate
is published without changing it or opening another holdout.

## Stage 1: causal development experiment

The fixed exposed population is the 152 prefixes already evaluated through V6.
No prefix exceeds 600 recorded seconds. The primary decision set is the six T2
sessions with greatest numeric acquisition day, chosen without new model scores.
All other exposed sessions remain secondary evidence, including T3. Raw tails and
new archive members are inaccessible to this stage.

All models use the existing causal combined spike-count and spike-band-power
representation with a 200 ms filter, ridge penalty 1, free intercept and unit-disk
prediction. Compare a frozen 120-second startup fit, expanding history, rolling
60/120/240-second fits, and exponentially weighted history with a 60-second half
life. These share a fixed first-30-second normalizer. A separately named rolling
120-second variant updates its normalizer on its own past training window.
This ablation separates changes to coefficients from changes to normalization.

Updates occur at 272, 332, 392, 452, 512 and 572 observed seconds. Labels stop two
seconds before each update. Predict the following 60 seconds, truncated at the
recording end or 600 seconds. Every method receives the same scoring mask, with
the 62-bin acquisition-block burn removed. At least 50 valid rows are required
for a scoring window. Short recordings and unavailable scores are retained.
Session R² pools predictions across eligible future windows. Windows are never
treated as independent replicates; bootstrap units are sessions for T2 and people
for cross-participant summaries. Trial IDs are absent from the prepared data,
so the embargo establishes temporal separation, not disjoint trials.

Dense labels are an explicit diagnostic privilege. Previously scored labels may
train later fits only after their fixed delay. Unique requested label time is
reported; repeated use of the same label does not incur duplicate acquisition
cost. These fits do not establish that a low-label policy can choose useful
updates. Frozen model exports and private full prediction traces support an
independent numerical audit.

## Gate and conditional continuation

An eligible rolling or exponential method must score all six primary sessions,
achieve mean R² at least 0.05, gain at least 0.05 over frozen120, have at least four
positive session scores, and have a positive lower endpoint of the paired
session-bootstrap 95% interval. All five eligible alternatives and failures are
retained. If multiple methods pass, choose the highest mean T2 R², breaking ties
by their declared order. This is exposed development selection; the intervals
do not correct for multiplicity and are not fresh confirmatory evidence.

If no method passes, stop before acquiring fresh data or constructing another
controller. That outcome limits this representation and update family; it does
not prove that T2 lacks neural information. Publish the diagnosis and the failed
gate. If it passes, write and publish a separate policy and acquisition lock,
compare periodic and triggered recalibration at matched total label budgets,
then assess unopened fixed sessions. The new policy must account for monitoring
labels, rejected updates, interruption time and failures before detection.

The [machine-readable lock](recovery-plan.json) is authoritative for numeric
choices. Code and source identities will be committed before any real recovery
run. Public artifacts contain aggregates and provenance; sample arrays, models
and prediction traces remain in ignored local directories.
