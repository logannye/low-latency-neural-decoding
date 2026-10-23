# V7B: accumulating updates and sparse feedback

## Explicit change to the continuation plan

V7A's declared recency gate **failed**. Its best eligible rolling240 fit had
mean T2 R² 0.049538, below the 0.05 floor; rounding does not turn this into a pass.
The ineligible expanding-history control reached 0.068916 and improved every
primary session, though one remained negative. These are exposed development
results, not fresh evidence.

The original V7A protocol said to stop after a failed gate. V7B is an explicitly
documented, post-development change to that stopping rule, made to pursue the
user's request to test continual recalibration if additional labels help. It is
not represented as a V7A gate pass. The hypothesis changes from emphasizing recent
labels to **retaining all acquired labels and placing feedback in time**. V7A's
plan, code, audited failure and source identities remain unchanged. This new
plan is committed before any V7B fresh outcomes are prepared or scored.

## Fixed experiment

Five procedures share the exact 152–600-second scoring interval: zero output,
frozen120, fixed240, periodic240 and triggered240. They retain short recordings
and count calibration periods and update delays as zero predictions. This is
an offline interruption assumption, not a measured change in human behavior.

Continual procedures start with 120 label seconds and may request four further
30-second bursts at six predetermined opportunities. Both spend 240 seconds on
complete recordings. Periodic requests follow a fixed schedule. Triggered
requests use a past-only neural distribution shift score, plus forced scheduled
probes that ensure the fixed budget is used. The trigger never sees target
directions, valid-go masks or scorer errors. All purchased labels train each
subsequent ridge fit; normalization stays fixed to the first 30 seconds.

Four fixed trigger thresholds are ranked on exposed development sessions, with
the target participant entirely excluded and all source results retained. The
audited threshold choices are committed before fresh assessment. T2 informed the
design, so this is not untouched-person development. New archive ranks19–21 in
the eight known participants provide the fresh-session test, with no substitution
for missing, short or difficult recordings.

Report full-interval prediction quality, time in windows worse than zero, label
use, interruption duration and every participant result. Periodic recovery and
the additional value of triggered timing have separate gates. A trigger is useful
only if it improves on periodic requests at the same feedback budget. A
distribution shift is a candidate request signal, not evidence of harm by itself.

The [full numerical plan](continual-plan.json) defines every interval, objective,
threshold and gate; the [acquisition lock](acquisition-plan.json) bounds data
selection and transfer. The historical failure remains in
[the V7A protocol](PROTOCOL.md), and all changes after its outcomes are disclosed
here rather than silently editing the original stopping rule.
