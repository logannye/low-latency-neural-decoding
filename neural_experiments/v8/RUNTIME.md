# V8 runtime measurement

This separate engineering check measures the portable decoder selected by the
V8 development study. It does not change the scientific producer, selection,
fresh cohort, or accuracy gates. The [runtime plan](runtime-plan.json) and
[implementation](runtime.py) are committed before timing begins.

The representative is the first T2 session in the factorial development split,
archive rank 16, using the selected primary model already exported by that run.
Two 60-second replays process the first 3,000 observations at 50 Hz. Each replay
warms the interpreter with 500 steps, resets filter state, and preserves the
recorded acquisition-block resets. The benchmark retains every prediction and
raw release/start/completion/deadline clock locally, including deadline misses.
No labels or behavioral context enter runtime prediction.

The model was fitted using labels through 150 seconds, which are later than these
replay inputs. Consequently, parity over this interval is a numerical check,
not causal accuracy evidence. The service timing includes the feature-row
dictionary, block reset, input validation, transform, filters, normalization,
linear head and unit-disk projection. Release-to-completion timing additionally
includes ordinary operating-system scheduling delay. Input acquisition,
upstream feature extraction, transport, file loading and fitting are excluded.
The two finite runs cannot establish a hard realtime guarantee.

The independent audit passed for both runs and all 6,000 updates. Public
[runtime aggregates](../results/v8-runtime.json) contain the exact private
manifest and audit identities. The representative is T2 day 237, with the
development-selected baseline configuration: TX4.5 plus spike-band power,
200 ms causal smoothing, ridge 1, and 120 seconds of calibration labels.

| Measured quantity | First run | Second run |
|---|---:|---:|
| Updates | 3,000 | 3,000 |
| Service median | 0.282 ms | 0.260 ms |
| Service p95 | 0.609 ms | 0.600 ms |
| Service p99 | 0.925 ms | 0.941 ms |
| Service maximum | 2.925 ms | 4.707 ms |
| Release-to-completion median | 3.620 ms | 3.912 ms |
| Release-to-completion p95 | 4.934 ms | 4.931 ms |
| Release-to-completion p99 | 5.377 ms | 5.290 ms |
| Release-to-completion maximum | 9.182 ms | 8.506 ms |
| Missed 20 ms deadlines | 0 | 0 |
| Process CPU consumed | 0.895 s | 0.831 s |

The decoder coefficients and normalizers occupy **6,160 bytes**; persistent
filter state and its decay coefficient occupy **1,544 bytes**. The compressed
model artifact is 6,268 bytes. These are numeric component sizes, excluding
Python object overhead, scratch arrays and the input cache. The model has
96 input channels and 192 filtered features.

Both streaming replays agreed with the exported batch decoder to a maximum
absolute error of 9.08×10⁻¹⁵. A separately implemented recurrence agreed with
the measured predictions to 7.77×10⁻¹⁶. The auditor independently reconstructed
all service and scheduling summaries from the raw clocks, checked CPU elapsed
time arithmetic and verified the numeric storage counts.

Measurement used Python 3.12.13 and NumPy 2.5.2 on macOS 26.2 arm64. The
benchmark requested a one-thread library limit and set the OpenBLAS/OpenMP
thread environment variables to one. `threadpoolctl` reported no detected
thread pools, so it did not independently verify effective library threading.
Other agents paused tests, fitting and audit workloads during timing; a
background dataset download continued. Its timing impact was not isolated.

The first measured attempt is retained. An earlier command supplied an incorrect
plan digest and was rejected before any workload was loaded, output directory
created, or timing recorded; its failure log is preserved. No latency outcome
was used to select a rerun or a different model. These operational annotations
are separate from the numerical audit.

The sub-millisecond service p99 describes the computation after inputs are
ready. The larger release-to-completion values include scheduling delay; neither
quantity measures acquisition-to-control latency or the temporal smoothing
delay. The 200 ms filter has exponentially decaying support and therefore adds
signal response dynamics independently of the measured CPU time.

Reproduce the same bounded measurement after rebuilding the audited source
artifacts, using a new output directory to preserve the original attempt:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python \
  -m neural_experiments.v8.runtime \
  --source results/neural_v8/development/manifest.json \
  --index data/neural_v8/development/index.json \
  --audit results/neural_v8/development-audit.json \
  --output results/neural_v8/runtime-reproduction \
  --plan-sha256 fc18ce69922721c7388477175f9236562d323fc25c50a0460758a5db6fe6bb78
```
