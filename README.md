# Low-Latency Neural Decoding

[![Neural decoding](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/neural.yml/badge.svg)](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/neural.yml)
[![Python tests](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/test.yml/badge.svg)](https://github.com/logannye/low-latency-neural-decoding/actions/workflows/test.yml)

Causal movement decoding under recording drift and runtime constraints. This
project tests how compact neural decoders generalize across recording days, how
adaptation can help or fail, and how much latency can be removed while preserving
exactly the same model. Quality, numerical correctness, and runtime have separate
measurements.

The primary experiment uses public FALCON H1 recordings: 176 neural channels,
20 ms bins, seven movement outputs, and one participant. Models consume current
and past neural observations. Evaluation targets and masks remain in the scorer.
The repository includes a causal FIR/ridge baseline, a 23,367-parameter TCN,
controlled adaptation tests, streaming inference, and independent numerical audits.

## Results and evidence

The completed initial development study averages R² **0.1607** for the TCN across
three seeds and three forward-day folds, versus **0.0526** for selected ridge.
S2 is a failure case. These development scores were used for selection and are
not final evidence. Additional experiments and sealed assessments are in progress;
the report records each completed gate and every negative result.

- [Study report](neural_experiments/REPORT.md), [protocol](neural_experiments/PROTOCOL.md), and [decisions](neural_experiments/NOTEBOOK.md)
- [Adaptation experiment](neural_experiments/ADAPTATION.md) and [runtime protocol](neural_experiments/RUNTIME_PROTOCOL.md)
- [Data provenance and use terms](neural_experiments/data/README.md)
- [Reproduction instructions](neural_experiments/README.md)

H1 measurements here are **local evaluations on public calibration recordings**,
not official FALCON leaderboard scores. Recorded-data replay measures processing
from counts being available; it does not establish live neural control, acquisition
latency, cross-participant generalization, implant power savings, or clinical benefit.

## Quick start

Use Python 3.12 and prebuilt scientific libraries:

```sh
python -m pip install --only-binary=:all: -e '.[dev,neural,runtime]'
export PYTHONPATH="$PWD/src"
python -m pytest -q tests/neural_decoder
neural-decode --help
```

Synthetic tests require no neural-data download. Follow the separate
[real-data reproduction steps](neural_experiments/README.md) after reading the
provider's data-use terms. Raw recordings, prepared arrays, model artifacts,
prediction traces, and raw timing records remain in ignored local caches. Public
artifacts contain code, protocols, metadata, aggregate results, and audit logic.
No Rust or other native extension build is needed for the neural experiment.

## Earlier work

The original CNN/Python/ARM64 experiment remains available in
[CNN_EXPERIMENT.md](CNN_EXPERIMENT.md). Its source, commands, and
[matched results](experiments/baselines/cifar10-matched-cpu-25-v1/README.md) are
preserved. Those measurements belong to that image-classification experiment;
they do not establish a neural-decoder speedup. The Python package is now named
`low-latency-neural-decoding`; existing `cnn-*` commands remain available.
