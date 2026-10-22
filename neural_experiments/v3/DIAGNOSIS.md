# Why the exposed V2 transfer models failed

This is a post-hoc diagnosis of already exposed V2 observations, not a new
held-out result. The executable [diagnosis](diagnosis.py) performs no fitting
and no decoder inference. It reads saved predictions from the original
120-second calibration models and examines only each selected session's first
600 seconds of observed bins. It never scores the later portions of those
sessions. The [aggregate receipt](../results/v3-diagnosis.json) binds the old
manifests, executed source snapshots, all original model identities, input
artifacts, and the 702 prediction traces used here by SHA256.

The diagnosis covers 117 models: deterministic ridge once per person and three
seeds for scratch, learned frozen core, full fine-tuning, and the post-hoc
random frozen core. It retains the complete original inventory at all three
budgets and all ten unavailable T6/300-second cases. Only the 120-second budget
is decomposed here. There are nine people, five future sessions per person,
and eight eligible common same-session suffixes. Metrics average days and seeds
within each person, then people. They do not weight long recordings more heavily.
All V2 artifacts remain unchanged.

## Fitting the prefix was not the main obstacle

| Method | Calibration-row R² | Same-session suffix R² | Future-day R² | Future centered correlation |
| --- | ---: | ---: | ---: | ---: |
| Ridge | 0.4986 | −0.3722 | −4.0407 | 0.0482 |
| Scratch | 0.7268 | −0.2474 | −3.1084 | 0.0669 |
| Learned frozen core | 0.5825 | −0.0793 | −1.5907 | 0.0584 |
| Full fine-tuning | 0.7785 | −0.3148 | −3.5462 | 0.0563 |
| Random frozen core | 0.3436 | 0.0230 | −0.5623 | 0.0596 |

Calibration-row scores are in-sample diagnostics, not validation accuracy. They
use the same 62-bin per-block training exclusion as V2. The common same-session
suffix starts at 300 seconds. Correlation centers each output within a scored
trace, sums the coordinate covariance, and divides by the two centered vector
norms; it is not an angular score.

The learned core's centered correlation falls from 0.7525 on calibration rows
to 0.4181 on the same-session suffix and 0.0584 on later days. The random core
generalizes better in squared error mainly while producing smaller and less
biased outputs; its future correlation is similarly weak. These observations
do not support interpreting its relative win as successful direction decoding.
They also make a simple shortage of optimization steps an unlikely sole cause.

## Unbounded outputs and bias amplify the failure

The targets lie on the unit circle, but every V2 model has an unconstrained
linear output. A fixed Euclidean projection onto the unit disk cannot increase
squared error against a target inside that disk. This is a deterministic
feasibility check, with no fitted parameter or selected threshold.

| Method | Future predictions outside unit disk | Future R² after fixed projection | Mean bias fraction of error | R² after diagnostic error-mean subtraction |
| --- | ---: | ---: | ---: | ---: |
| Ridge | 58.00% | −0.6936 | 37.30% | −1.3568 |
| Scratch | 64.54% | −0.7600 | 42.72% | −0.8414 |
| Learned frozen core | 44.04% | −0.6213 | 34.07% | −0.4078 |
| Full fine-tuning | 70.35% | −0.8317 | 44.47% | −0.9958 |
| Random frozen core | 17.17% | −0.3791 | 17.14% | −0.2170 |

Error-mean subtraction uses evaluation labels and is **not a deployable
correction**. It is the exact decomposition
`SSE = N * ||mean(prediction − target)||² + centered_error_SSE`.
Bias fractions are averaged per trace rather than computed from globally
pooled error. Allowing a separate error mean for every recording block raises
the learned core's bias fraction only from 34.07% to 35.64%.

Both checks leave poor absolute scores. The fixed zero-vector predictor has
mean future-day R² −0.0176, better than every model and every projected model
above. A zero vector provides no useful direction command. It is a necessary
sanity baseline here, not a successful decoder. Neither shrinking output
magnitude nor removing bias establishes recovery of the neural–target mapping.

## Frozen normalization meets a changing feature distribution

V2 uses square-root counts and freezes channel means and standard deviations
from the earliest session's permitted calibration prefix. Source training uses
source-only statistics. Source inspection found no inconsistent application of
these stored transforms. The concern is their suitability across sessions.

Across the nine calibration-day windows, the RMS channel-mean displacement in
calibration-standard-deviation units averages 0.073. Across 45 future-day
windows it averages 0.719. The fraction of neural observations with absolute
standardized value above five rises from 0.41% to 2.16%. On average, 52.43% of
channels hit the fixed standard-deviation floor of 0.1 in the 120-second
calibration prefix. Sparse channels can consequently have large standardized
excursions when activity changes.

The retained electrode-ID hashes are stable across each person's six sessions.
This checks ordering metadata, not identity of the underlying neural population.
The feature shift and output failures are consistent with inter-session drift,
but they do not isolate biological drift from behavior coverage, recording
conditions, or the effects of normalization. Whole-future-session centering
would use future observations and is not an admissible real-time fix.

## Pretraining already failed its own source-day generalization check

Source optimization loss decreases, but source validation usually deteriorates.
The equal-person validation MSE averages 0.5773 at 600 steps, 0.6709 at 1,200,
and 0.6691 at 2,400. For unit-direction targets the zero-vector MSE is 0.5.
At the first checkpoint, 25 of 27 source fits already exceed that baseline;
22 of 27 select this earliest checkpoint. Mean training loss over the first
80 steps is 0.5182, falling to 0.3173 over the final 80 steps.

Thus the source core cannot be assumed to supply robust temporal features.
Source checkpoint selection used MSE, while the headline endpoint averages
per-day R². This objective distinction should be explicit in any replacement
protocol, although it alone does not explain validation MSE worse than zero.
Additional source optimization or a larger temporal network is not the first
repair supported by these observations.

## Endpoint validity remains a separate issue

The provider's [pinned decoding analysis](https://github.com/nptl-stanford/array-paper/blob/93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834/array_paper/metrics/decoding_snr.py)
uses 300 ms averaged neural and position-error vectors, selected trial-onset
reaction-time windows, and near-target exclusions. The reaction-time rule gates
trial windows; it is not a neural–label temporal shift. Its source fitting uses
position error with trial-level affine calibration. V2 instead directly fits
instantaneous 20 ms unit cursor-to-target direction across valid go intervals.
The analyses therefore do not share an endpoint or evaluation protocol.

The companion raw-data audit reconstructed the exposed labels, masks, and
within-person channel metadata without finding a sign, transpose, interval,
or channel-order error. The contemporaneous target remains a closed-loop
intention proxy, not observed limb velocity or independent intention ground
truth. Provider whole-block centering must not be copied into a causal decoder
without replacing it with a permitted prefix or past-only procedure.

## Consequences for the next fixed protocol

The highest-priority repair is a strongly regularized causal linear decoder
with an explicit unit-disk output bound and a separately declared daily
calibration condition. Daily recalibration changes the information budget and
scientific question; it must not be presented as repairing V2's fixed
earliest-day, no-further-labels transfer claim. The old condition remains a
separate negative result.

Normalization, smoothing, and penalty selection should use source participants
and exposed development windows only. A new session tail can test a locked
recipe on fresh observations, but these participants have already contributed
to diagnosis and are not untouched participant-level test subjects. Useful
accuracy requires an absolute improvement over declared simple baselines,
not just a gain over an even more negative network. The evidence does not
justify another neural fit before those protocol and baseline checks.

## Reproduction

From the repository root, with the existing scientific Python environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python neural_experiments/v3/diagnosis.py \
  --index data/neural_v2/index.json \
  --run results/neural-v2/transfer-v1 \
  --control results/neural-v2/random-core-v1 \
  --output work/v3-diagnosis/reproduced.json
PYTHONPATH=.:src python -m pytest -q tests/neural_decoder/test_v3_diagnosis.py
```

The output includes aggregate numeric diagnostics and identities, with no
sample arrays or private absolute paths. Tests check constant predictions,
pure bias, blockwise bias, unit-ball projection, scoring masks, and equal-person
aggregation. They validate decomposition arithmetic, not the original training
process or the biological interpretation.
