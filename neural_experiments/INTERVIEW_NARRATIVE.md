# One-page project narrative

I wanted to test two practical constraints together: whether a small neural decoder
could remain useful on later recording days without new labels, and whether its
streaming implementation could finish reliably within a 20 ms observation period.
I used public FALCON H1 recordings from one participant, with 176 neural channels
and seven movement outputs. I kept model quality, numerical equivalence, and
processing latency as separate claims.

I began with a constant predictor and a causal filtered ridge decoder, then used
one compact TCN with 23,367 parameters and a 300 ms causal receptive field. The
important engineering boundary was information availability: training transforms
were frozen, inference received counts only, trials did not reset state, and each
provider file began from reset. A manifest gate kept later-day arrays unavailable
until source, recipes, weights and metrics were frozen. Independent scoring used
complete files grouped by day, retaining negative R² and every seed.

The first research idea did not survive its own decision rule. I tested bounded
mean correction against ordinary causal recentering and frozen normalization.
Recentring improved the development average, but the movement-mixture diagnostic
did not expose the predicted harm that a bound was supposed to prevent. I retained
that positive observation and the failed protective rationale, and kept frozen
normalization for the final model. One final training intervention, fixed 10%
channel dropout, improved development mean R² from 0.1607 to 0.1722 and passed the
prespecified day-harm rule. I used all three seeds, not the strongest one.

After freezing, the TCN averaged R² 0.2482 on S5 versus 0.1625 for ridge. I then
refitted the unchanged recipe on S0–S5 and evaluated seven untouched later days.
The TCN averaged 0.1776 versus 0.0341 for ridge, an absolute difference of 0.1434.
Every seed/day combination beat ridge, although the `tz` output remained weak.
The later evidence is only about 11.5 minutes of released recordings across seven
days in one participant. I would not generalize it to other participants, clinical
benefit, or closed-loop control.

For the systems comparison I held weights, precision, inputs, and thread budget
fixed. I compared PyTorch sliding-window execution, incremental NumPy, and an
explicit-state ONNX graph. On an M4 Max, NumPy's median fast-replay p99 was 0.0343 ms
versus 0.1075 ms for PyTorch, a 3.13× ratio. Paced replay changed the picture:
NumPy's p99 rose to 0.428–0.538 ms, and host scheduling added roughly 5 ms before
completion. ONNX had one 31 ms service-time outlier and a missed deadline. All
repeats, raw timestamps, queue delays and misses are retained. Those measurements
use a separately pinned development weight file; final-model quality uses the
frozen refits.

The lesson was to optimize the complete measured stream and keep the evidence
boundary visible. A favorable kernel number does not establish paced behavior;
a developmental adaptation gain does not establish its mechanism; and a hash
proves artifact identity rather than scientific truth. The next useful evidence
would be a larger independent participant/day study and an actual acquisition-to-
action deployment test. The current repository supplies the code, protocols,
aggregate evidence, private replay generator, independent audits and hosted tests
needed to inspect and reproduce this narrower result.
