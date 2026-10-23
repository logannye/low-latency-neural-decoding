# V7B descriptive monitoring diagnostics

These definitions are locked before fresh scoring. They describe the already
frozen policy; they never select thresholds, change requests, or alter the
primary success gates. The evaluator can use targets that the controller cannot.

**Recovery delay:** use the declared consecutive 30-second scoring windows.
An episode begins at the start of the first window with summed MSE strictly
greater than zero output (tolerance 1e-12). It ends at the end of the first later
window strictly better than zero. Neutral windows, including zero output during
calibration, do not constitute recovery. A further bad window continues the same
episode. Windows with fewer than 50 valid rows do not establish either state;
their presence during an episode is recorded. Episodes still open at the observed
recording end are right-censored. Report every episode, resolved and censored
counts, and delay among resolved episodes separately. This coarse retrospective
latency is not the exact onset of biological drift or proof that an update caused
recovery. Using the end of the recovery window includes the observation time
needed to establish its result.

**False-alarm proxy:** at each triggered-policy decision, compare the actual
policy predictions with zero output over the preceding 60 seconds, truncated at
the common 152-second deployment boundary. Use valid scoring rows outside
calibration interruptions and require at least 50. A request explicitly caused
by the neural-drift threshold is a proxy false alarm when that preceding active
interval was no worse than zero (MSE difference <=1e-12). Forced quota/gap requests
are reported separately. Also report below-threshold decisions preceded by worse
than zero performance. Neither diagnostic establishes the causal value of labels:
apparently unnecessary requests may prevent later failures, and requests after
poor predictions may not repair them. No window-level independence claim is made.

All session-method outcomes and decision dispositions are exported as aggregate
numbers. Sample predictions and targets remain in ignored local artifacts.

## Pre-assessment numerical correction

Before fresh scoring, a synthetic unit-circle test found that float32 targets
and float64 predictions could give a nonzero error difference for exact zero
output: the residual and baseline terms were squared at different precisions.
Both selected arrays are now promoted to float64 before either calculation in
the producer and independent verifier. Regression tests require exact zero
difference and the corresponding neutral classification. The diagnostic
definition, tolerance, policy, primary scoring and gates are unchanged. Original
auditor snapshots remain retained alongside the corrected version.
