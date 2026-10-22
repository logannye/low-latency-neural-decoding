# Decisions and observations

## 2026-10-22: starting gate

Verified clean repository at the handoff revision. Created branch
`neural-decoder/h1-baselines`; preserved the image experiment. Apple M4 Max,
128 GB RAM, arm64 macOS 26.2; approximately 329 GiB available disk space.
Use existing Python 3.12 environment with NumPy/SciPy/sklearn/PyTorch; added
only prebuilt HDF5/PyNWB dependencies. No local native compilation.

H1 live inventory remains 40 files / 102,258,934 bytes. Draft is mutable and has
provider metadata validation issues, so complete asset identities are recorded.
Provider CC-BY-4.0 metadata accompanies an explicit no-redistribution data-use
agreement. Keep raw recordings and derived sample-level arrays ignored/local.

The physical hypothesis is that session-dependent observations can change while
the behavior-related mapping is partly stable. Activity-distribution change alone
cannot distinguish a recording nuisance from a movement-mixture change. No such
mechanism has yet been established for these recordings.

The first six-configuration ridge grid was fixed before scores. It compares a
short versus longer causal filtered history. This is a development exercise;
sealed S5 and S6-S12 remain unavailable to fitting and exploratory scoring.

## 2026-10-22: G1/G2 results and first research hypothesis

The six-setting ridge grid selected history 1, lambda 0.1. Its day scores are
-0.072894, 0.099554, 0.131214 (S2/S3/S4). A single fixed TCN recipe averaged 0.160722
across three seed-level means, versus 0.052625 for ridge. TCN S2 scores range from
-0.153394 to 0.012743. All values and seeds are retained; no best seed was selected.
Independent audits reproduced the scores and inference traces. Final days remain
sealed. G1/G2 are implemented and locally evaluated; later gates are unfinished.

1. **Observation:** S2 is poor for both families. Selected ridge residual mean
   explains 18.19% of its total squared error, especially output g3. Yet S3 has a
   larger aggregate neural mean shift and better predictions. Mean shift alone
   is not a sufficient detector of decoding harm.
2. **Mechanism hypothesis:** Part of the S2 prediction bias could arise from
   observation offsets along decoder-sensitive feature directions. A shift in
   movement mixture could create the same marginal observation shift, so freely
   recentering the stream could also remove informative activity.
3. **Prediction:** A small source-anchored causal mean correction should reduce
   bias under an imposed observation offset and may improve real development-day
   scores. A bounded correction should limit harm relative to unrestricted
   recentering when a different movement mixture changes the input mean.
4. **Discriminating experiment:** Keep the fitted ridge weights and variance
   scale fixed. Compare frozen centering, an ordinary causal EMA mean correction,
   and bounded versions with caps 0.25/0.5/1.0 training SD. Use a fixed 30-second
   time constant, initialize correction at zero, predict before updating, and
   apply the same state reset rules. The candidate state equation is
   `m_next = (1-alpha)*m + alpha*(x_filtered-mu_train)`, with
   `alpha=1-exp(-0.02/30)`; prediction subtracts either zero, previous m, or
   `clip(previous_m, -cap*sigma_train, cap*sigma_train)`. This first test isolates
   mean correction; variance adaptation is deliberately deferred. Assess a
   controlled observation-offset stress and a movement-mixture stress using
   development recordings only. Before execution, freeze the stress magnitude,
   contiguous interval construction, history/reset semantics and scoring rules;
   labels may construct retrospective diagnostic groups but never enter predictor
   state or trigger adaptation. Keep synthetic stress evidence separate from
   real forward-day performance.
5. **Decision rule:** Retain a bounded correction only if it improves mean
   development R² by at least 0.005, harms no day by more than 0.01, and improves
   the observation-offset test while limiting mixture-stress harm relative to
   unrestricted correction. Report the number of days harmed and worst harm.
   These are development decision thresholds, not biological or clinical
   requirements. If no setting qualifies, retain frozen normalization and report
   the negative result; do not expand the algorithm search to find a headline.

No adaptive experiment has been run. This is the first mechanism hypothesis,
not evidence that recording offset caused the failure. The protocol and numerical
stress construction will be locked before G3 execution.

## Integration notes

A source-asset mapping gate now prevents sealed files from being relabeled as
allowed development days. A replay sequence-ID check prevents fractional IDs
from being silently truncated. Independent audits verified that the already
completed runs used the correct inputs. The run snapshots retain their original
code; these fixes do not rewrite prior evidence.

The local macOS environment marks editable-install .pth files hidden, so Python
may ignore them. Reproduction commands explicitly set PYTHONPATH to the source
folder; this applies to both the old classifier and the new neural package.
The installed entry point is implemented, and hosted Linux packaging is validated
separately by CI. No native build is needed.

The public repository is named computational-experiments. The original CNN
README is preserved in CNN_EXPERIMENT.md; the public root README introduces both
tracks. Raw/prepared data, checkpoints, per-bin predictions, and the private
planning handoff remain local. Publishing code does not constitute completion
of adaptation, runtime measurement or sealed assessment.

## 2026-10-22: G3 and final development decision

The repository was renamed to `low-latency-neural-decoding` at the user's request.
The root README and distribution now focus on neural decoding; historical CNN
source, commands and evidence are preserved. A new checkout reuses the existing
read-only H1 cache and Python interpreter; new runs use fresh local directories.
Only prebuilt ONNX/ORT wheels were added. No local native compilation occurred.

Bounded correction did not pass its locked mixture-harm advantage criterion.
Ordinary EMA improved real mean R2 from 0.052625 to 0.142009 and harmed no
development day, but the diagnostic also harmed no day, leaving no demonstrated
protective advantage for a bound. Retain frozen normalization; do not turn the
unmet criterion into a post hoc positive claim. All 90 traces and 45 day scores
passed independent arithmetic/scoring audit.

The second and final hypothesis, training channel dropout p=0.1, improved
three-seed mean development R2 from 0.160722 to 0.172153. Seed-average day gains
were 0.010648/0.005737/0.017909, meeting both prespecified retention conditions.
Some individual seed/day differences are negative. Retain this single new
configuration with all three seeds. Nine model/18-trace independent audits passed.
The nonlinear family used two configurations total; no further architecture,
epoch, dropout probability, adaptation time constant or stress setting is searched.

The final recipe, metrics and separate calibration budgets are recorded before
S5 access. S0-S4 final refits are complete; array release awaits the exact
committed-source freeze and runtime validation. S5-S12 remain sealed at this entry.


## 2026-10-22: completed frozen evidence and stretch decisions

This entry supersedes earlier progress notes that describe final data as sealed
or runtime measurements as pending. The early freeze was recorded at 8d7220e,
then S5 was released and scored once: ridge 0.162497, three-seed TCN mean 0.248215.
The identical recipe was refitted on S0–S5 and frozen at 4f30a3d with unchanged
implementation hashes. Across S6–S12, ridge averaged 0.034146 and TCN 0.177583.
All 21 TCN seed/day scores exceeded ridge; `tz` remained weak. Independent audits
reconstructed all 80 final traces and every day/output score. No final result
triggered model selection or changed the primary recipe.

Same-weight runtime measurement used the prespecified original seed 0/S4 artifact,
separately from the final dropout refits. Three fast repeats and two 60-second
paced repeats per implementation were measured sequentially on M4 Max. Incremental
NumPy's median fast p99 was 0.034333 ms versus 0.107541 ms for PyTorch. Its paced
p99 was 0.428334–0.537959 ms with zero observed deadline misses. ONNX had one
31.224541 ms service outlier and a deadline miss; no sample was removed. The
independent audit checked all 15 timing blocks and prediction traces. This is
counts-ready recorded replay, not acquisition-to-action or closed-loop evidence.

The separate intercept calibration study used labels within fixed 0/5/15/30-second
file prefixes and scored one common suffix. It helped ridge modestly but did not
improve the TCN average at any budget. All 224 calibration traces were independently
reconstructed exactly. Runtime snapshot/restore was implemented and tested;
full training resume and new native kernels were unnecessary for this scope.

Dynamic int8 quantization passed its nine-model development quality gate (mean
R² change +0.001731; worst -0.000617), reduced graph bytes by 61.47%, and lowered
fast p99 about 2.15× in a separate float32/int8 campaign. Paced rankings reversed
between repeats, so no consistent paced advantage is claimed. Float32 final
models remain primary; final-day compressed accuracy was not evaluated.

The prespecified M2 minival transfer check completed within its storage cap.
Four training trials supplied only 302 bins (6.04 seconds); evaluation covered
three later days. Ridge mean R² -13.497285 was much worse than constant -0.034258.
Independent preparation/refitting/scoring confirmed this failure. We retain it
as evidence of numerical portability and failed quality transfer under sparse
training, and do not retune or generalize H1 success to a second participant/task.

A private seven-output replay was generated from a completed later-day run and
visually checked. The public repository includes its generator, aggregate plots,
audit receipts, reproduction guide and final report. Raw arrays, trained artifacts,
sample traces and the private replay stay local. Hidden EvalAI evaluation was not
submitted because local authentication is unavailable; there is no official score.
