# Independent review of the first compact TCN study

`results/neural-h1/tcn-dev-v1` passed an independent inference and scoring audit
on October 22, 2026. All nine model artifacts and 18 full recording traces were
checked, covering three predetermined seeds on the three forward development
folds. No additional model was trained. No S5 or later-day sample arrays were
opened, and no sealed result is reported.

| Seed | S2 R² | S3 R² | S4 R² | Equal-day mean R² | Population SD across days |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.153394 | 0.283955 | 0.316845 | 0.149135 | 0.214341 |
| 1 | -0.036288 | 0.216313 | 0.231170 | 0.137065 | 0.122729 |
| 2 | 0.012743 | 0.267486 | 0.307665 | 0.195964 | 0.130591 |

The mean of the three seed means is **0.160722**, and their population standard
deviation is **0.025403**. The selected ridge development mean was 0.052625. This
comparison uses the same day splits and scored rows, but these days are
development data and the TCN was studied after reviewing baseline failures. The
nine seed/day values are not nine independent biological recording days. Seed
variation is shown separately from across-day variability; neither standard
deviation is a confidence interval.

The TCN improves S3 and S4 in every seed relative to the selected ridge. S2
remains unstable: seed 0 is worse than the ridge, seed 1 is better but negative,
and seed 2 is slightly positive. No best seed has been selected to replace the
full result table. No general cross-session improvement or adaptation benefit
is established by these development observations alone.

## Independent reconstruction

`neural_experiments/audit_temporal.py` loads the numeric NPZ weights and constructs
a whole-file calculation with `torch.nn.functional.conv1d`, left padding, ReLU,
and a final linear projection. It imports neither the production `CausalTCN`
class nor `TCNStream`. Each layer processes the full input file, so this audit
does not reuse incremental rings, layer caches, or the production sliding-window
reference. Frozen preprocessing and output conversion use the saved float32
arrays, matching the measured inference contract.

Across all 18 complete traces, the maximum absolute difference from the saved
incremental predictions was **2.7939677e-08** in target units. The largest
absolute difference in resulting day R² was **1.1939825e-07**. The audit retained
all file-level parity values and all day/output scores. Per-output R² and
variance-weighted day R² were recomputed independently with scikit-learn using
the declared float64 scoring calculation; saved aggregate and per-output scores
agreed within 1e-12 absolute tolerance. Original float32 predictions were
retained, not refitted or replaced.

The manifest, configuration, source snapshot and diff, prepared input index,
all development feature/target files, models, and saved prediction hashes were
checked. The source authority check reconstructs file/day/provider roles through
`data.load_index` before any sample arrays are opened. Frozen input mean/scale
and target mean/scale were independently reproduced for each fold from its
authorized training files and matched exactly. The scoring labels and masks
were joined after the feature-only whole-file inference call. Neither influences
inference state or resets.

The complete evidence is `results/neural-h1/tcn-dev-v1/independent-audit.json`,
with matching `independent-audit-source.py`. Reproduce into a new file:

```sh
PYTHONPATH=src .venv/bin/python neural_experiments/audit_temporal.py \
  --run results/neural-h1/tcn-dev-v1 \
  --index data/neural_h1/prepared/index.json \
  --output results/neural-h1/tcn-dev-v1/independent-audit-new.json
```

## Reviewed training and causal contract

The model has **23,367 learned parameters**: three 32-channel causal
convolutions with kernel width 3 and dilations 1, 2, and 4, followed by a
seven-output linear head. Its receptive field is 15 bins, a 300 ms input
collection window at the nominal 20 ms period. It has no future lookahead.
Zero padding occurs in standardized input and hidden-activation space, and
each file starts from reset. Trials and scoring masks do not reset state.

The fixed recipe uses 20 epochs, Adam, learning rate 0.001, no weight decay,
and the final epoch checkpoint. Training examples have 14 real preceding bins
from the same file before the scored part of each chunk. The first 14 bins of
each training file are excluded from training loss, while evaluation predicts
every bin and follows the official mask. Training normalizers use only the
training fold; target normalizers use its supplied scored target rows. The
source has no evaluation-based checkpoint selection or early stopping.

This review verifies retained inference artifacts and their scores. It does not
independently rerun optimization or establish that another environment will
reproduce every learned bit. The recorded source is an exploratory snapshot
rooted at the historical Git revision, with the new source bytes retained by
hash; final measured comparisons need committed reproducible source.

Whole-file arithmetic parity is a correctness result. It does not demonstrate
a speedup, bounded p99 service time, deadline behavior, process memory, energy
savings, or completion of G4. Those require a separate fair runtime experiment
with identical weights, workload, precision, thread budget, and complete-step
measurement boundaries.
