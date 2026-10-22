# Computational Experiments

[![Python tests](https://github.com/logannye/computational-experiments/actions/workflows/test.yml/badge.svg)](https://github.com/logannye/computational-experiments/actions/workflows/test.yml)
[![Neural decoding](https://github.com/logannye/computational-experiments/actions/workflows/neural.yml/badge.svg)](https://github.com/logannye/computational-experiments/actions/workflows/neural.yml)

Research and engineering experiments in learning, temporal signals, and efficient
computation. Each study starts with a concrete question, builds an inspectable
implementation, and measures it against explicit baselines. Code correctness,
model quality, and runtime performance have separate evidence.

## Experiments

| Study | Question | Current evidence |
|---|---|---|
| [Neural decoding across recording days](neural_experiments/README.md) | Can a compact causal decoder estimate movement on a new recording day without labeled recalibration? | H1 development study: TCN mean R² **0.161**, selected ridge **0.053**, three forward-day folds and three TCN seeds. One day remains unstable. Final days are sealed. |
| [CNN: Python vs. ARM64 assembly](CNN_EXPERIMENT.md) | What changes when the same small CNN is trained and executed through different implementations? | Historical matched 12-run study: assembly training **1.69× faster**; both implementations observed about **51.99%** mean CIFAR-10 test accuracy on one M4 Max. |

These results belong to different tasks, source states, and measurement protocols.
The CNN speedup is not a neural-decoder speedup. H1 scores are local development
results on public calibration recordings, not official FALCON scores or untouched
final evaluations. Neither study establishes clinical performance or a general
advantage of one implementation strategy.

## Neural decoding

The new track uses FALCON H1: 176 neural channels, 20 ms bins, and seven movement
outputs from one participant. A causal FIR/ridge baseline and a 23,367-parameter
TCN use current and past observations. Feature-only streaming, file resets,
source/data manifests, causal intervention tests, and independent numerical audits
make the pipeline inspectable.

The TCN improves the development average, while S2 remains a failure case. The
[next hypothesis](neural_experiments/NOTEBOOK.md) examines whether a bounded
normalization correction can reduce prediction bias without removing useful
movement information. Adaptation, sustained latency/deadline measurements, and
sealed-day evaluation are future gates. The incremental implementation currently
has numerical parity evidence, not a measured speedup claim.

- [Development report and every seed/day result](neural_experiments/REPORT.md)
- [Protocol and information boundaries](neural_experiments/PROTOCOL.md)
- [Dataset identities, schema, and use terms](neural_experiments/data/README.md)
- [Independent ridge review](neural_experiments/BASELINE_REVIEW.md) and [TCN review](neural_experiments/TCN_REVIEW.md)
- [Setup and reproduction commands](neural_experiments/README.md)

## CNN and assembly

The original image experiment retains its source, native implementation, commands,
checkpoints, and evidence. Its complete former README is preserved as
[CNN_EXPERIMENT.md](CNN_EXPERIMENT.md), including architecture, installation,
training, evaluation, and reproduction instructions. The
[matched protocol](comparison/PROTOCOL.md) and
[full results](experiments/baselines/cifar10-matched-cpu-25-v1/README.md) remain at
their original paths.

## Working locally

Use Python 3.12 and prebuilt scientific dependencies. For the neural track:

```sh
python -m pip install --only-binary=:all: -e '.[dev,neural]'
export PYTHONPATH="$PWD/src"
python -m pytest -q tests/neural_decoder
neural-decode --help
```

Synthetic tests need no neural-data download. Acquire H1 only through the explicit
[reproduction steps](neural_experiments/README.md) after reading its data-use terms.
Raw recordings, prepared sample arrays, model artifacts, and prediction traces
remain in ignored local caches. The repository contains source, protocols,
metadata manifests, aggregate results, and independent audit code.

The Python distribution keeps its existing `cnn-python-vs-assembly` name to
preserve installed command compatibility; the repository is now
`logannye/computational-experiments`.
