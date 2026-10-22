# Compact temporal candidate

This is the one initial nonlinear candidate for G2, after the G1 baseline gate.
Its implementation and synthetic arithmetic tests do not establish decoding
quality. Retain the ridge decoder if it wins the development comparison.

The model maps a causal observation history to seven simultaneous outputs using
three single 1D convolution layers, each followed by ReLU. Their input/hidden
widths are 176 → 32 → 32 → 32, kernel width is 3, and dilations are 1, 2 and 4.
A pointwise linear head produces seven outputs. There are 23,367 trainable
parameters. The receptive field is `1 + 2*(1+2+4) = 15` bins: the present bin and
14 previous bins, spanning 280 ms between first and last bin centers and covering
300 ms of binned observations at the configured 20 ms interval. There is no
future context, time-aggregating normalization, dropout or residual block.

This is a deliberately compact extension of convolution experience. The same
architecture also makes repeated computation visible: recomputing a 15-bin
window updates hidden activations that an incremental implementation can cache.
The choice is a starting hypothesis, not a measured improvement over ridge.

## Training and inference contract

`CausalTCN.forward` accepts a float32 tensor of shape `[batch, time, 176]` after
training-derived feature standardization and returns standardized targets of
shape `[batch, time, 7]`. Fit every affine statistic using authorized training
data only, handle constant dimensions with an explicit positive scale, and
invert target scaling before scoring. `freeze_tcn` or `export_tcn` attaches
`input_mean`, `input_scale`, `target_mean` and `target_scale`. Omitted affine
terms are identity transforms for synthetic tests, not a new estimation rule.

Begin with masked mean squared error and fixed file-respecting training chunks.
For a loss interval `[start, stop)`, supply input beginning at
`max(0, start-14)`, and calculate the loss only on the designated loss interval.
The preceding 14 bins are context, not extra loss observations. This reproduces
the whole-file causal outputs on the loss interval exactly up to float32
arithmetic. At the first file bin there is no invented context. Trial markers
do not reset context. Never join the end of one file to the start of another.
The scoring mask may select training loss terms under the protocol, but must
never enter the predictor, alter feature history or stop consumption of bins.
Chunk length, optimizer, epochs, seed and selection rules belong in the run
configuration; no configuration is empirically selected by this module.

`TCNStream(frozen, implementation="incremental")` consumes one raw feature
vector with `step`, or an ordered chunk with `predict_chunk`. Both return target
units. `reset()` is required exactly at recording-file boundaries. Empty chunks
do nothing; ordinary chunk/trial boundaries preserve state. The numeric engine
accepts no labels, masks or trial IDs. Channel identity/order, count validity,
timestamps, sequence IDs and gap policy must be enforced by the outer replay
contract. Only finite numeric vectors of the fixed feature width are accepted.
An arithmetic failure invalidates the trace; reset before reusing that stream.

## Same-weight execution

The `window` mode recomputes a PyTorch forward on the last 15 *observed* bins.
During the first 14 bins its actual input window is shorter. This detail matters:
padding a raw 15-bin window before running the model would create hidden
activations from biases at nonexistent times and disagree with whole-file
left-padding. Missing history at each layer is zero activation in that layer's
space, not hypothetical prior zero-count observations.

The `incremental` mode stores each layer's input in a zero-initialized ring of
length `1 + 2*dilation`. For convolution tap `k`, it reads lag `(2-k)*dilation`,
matching PyTorch cross-correlation ordering, then adds bias and applies ReLU.
Layer outputs are computed for the current bin before downstream layers consume
them. Rings never grow with stream duration. Python scratch allocations remain;
neither implementation claims allocation-free execution or a latency bound.

Both modes use identical exported float32 weights and affine transforms. Tests
compare every prediction against whole-file PyTorch forward, including biased
startup, zero inputs, impulses and a deliberate 10,000-count stress. Tolerance is
`rtol=2e-5, atol=2e-5`, with `atol=0.002` only for that extreme-input stress.
This is arithmetic evidence, not a biological quality tolerance. Any speed
comparison must first validate complete real-data prediction traces and quality,
then use identical threads, hardware, stream order and timing boundaries. This
module does not substitute eager PyTorch for the required optimized standard
runtime comparison at G4.

## Inference artifact

The NPZ uses numeric arrays only and `allow_pickle=False`. Version 1 requires
exact names, dimensions and dtypes, finite float32 values, strictly positive
scales and a scalar int64 format version. Unexpected/duplicate ZIP entries and
oversized decompressed archives are rejected. Exports never overwrite an
existing path. This is an inference artifact, without optimizer, RNG, training
resume or portable runtime restore state. Bind its file hash, channel-order
identity and normalization-fit provenance in the surrounding run manifest;
the artifact's structural validation cannot establish those provenance facts.

Synthetic tests exercise full-forward causality, gradient receptive field,
future-prefix/state isolation, chunk partition invariance, trial continuity,
file reset, bounded state, 14-bin training burn-in equivalence, corrupt inputs,
corrupt artifacts, export/reload identity and complete cross-backend parity.
