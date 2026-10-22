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
