"""Synthetic arithmetic/causality evidence; none of these tests imply H1 accuracy."""

import numpy as np
import pytest
import torch

from neural_decoder.temporal import (
    CausalTCN,
    FrozenTCN,
    TCNStream,
    export_tcn,
    freeze_tcn,
    load_tcn,
)


@pytest.fixture(autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model_and_artifact():
    torch.manual_seed(109)
    model = CausalTCN().eval()
    # Large, unequal biases expose synthetic prehistory at the start of a file.
    with torch.no_grad():
        for i, conv in enumerate(model.convs):
            conv.bias.copy_(torch.linspace(-0.2, 0.7 + i * 0.1, 32))
    rng = np.random.default_rng(8)
    frozen = freeze_tcn(
        model,
        input_mean=rng.uniform(0.3, 3.0, 176),
        input_scale=rng.uniform(0.5, 2.0, 176),
        target_mean=np.linspace(-1, 1, 7),
        target_scale=np.linspace(0.5, 2, 7),
    )
    return model, frozen


def _full_trace(model, frozen, features):
    a = frozen.arrays
    standardized = (features.astype(np.float32) - a["input_mean"]) / a["input_scale"]
    with torch.inference_mode():
        prediction = model(torch.from_numpy(standardized)[None])[0].numpy()
    return prediction * a["target_scale"] + a["target_mean"]


def _assert_state_equal(left, right):
    assert left[:2] == right[:2]
    np.testing.assert_array_equal(left[2], right[2])
    for a, b in zip(left[3], right[3], strict=True):
        np.testing.assert_array_equal(a, b)


def test_architecture_receptive_field_and_gradients():
    torch.manual_seed(7)
    model = CausalTCN()
    assert sum(parameter.numel() for parameter in model.parameters()) == 23_367
    features = torch.randn(2, 39, 176, requires_grad=True)
    prediction = model(features)
    assert prediction.shape == (2, 39, 7)
    prediction[:, 27].sum().backward()
    assert torch.count_nonzero(features.grad[:, :13]) == 0
    assert torch.count_nonzero(features.grad[:, 28:]) == 0
    assert features.grad[:, 13].abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_full_forward_future_intervention(model_and_artifact):
    model, _ = model_and_artifact
    torch.manual_seed(5)
    first = torch.randn(1, 67, 176)
    second = first.clone()
    second[:, 29:] = 100 * torch.randn(1, 38, 176)
    with torch.inference_mode():
        torch.testing.assert_close(model(first)[:, :29], model(second)[:, :29], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["incremental", "window"])
def test_stream_prefix_and_state_are_future_independent(model_and_artifact, mode):
    _, artifact = model_and_artifact
    features = np.random.default_rng(88).poisson(2, (51, 176))
    changed = features.copy()
    changed[17:] += 50
    first, second = TCNStream(artifact, mode), TCNStream(artifact, mode)
    np.testing.assert_array_equal(
        first.predict_chunk(features[:17]), second.predict_chunk(changed[:17])
    )
    _assert_state_equal(first.state_snapshot(), second.state_snapshot())
    assert not np.array_equal(
        first.predict_chunk(features[17:]), second.predict_chunk(changed[17:])
    )


@pytest.mark.parametrize("kind", ["counts", "zeros", "large", "impulses"])
def test_complete_trace_export_and_incremental_parity(tmp_path, model_and_artifact, kind):
    model, frozen = model_and_artifact
    rng = np.random.default_rng(90)
    features = rng.poisson(3, (97, 176)).astype(np.float32)
    if kind == "zeros":
        features.fill(0)
    elif kind == "large":
        features = rng.uniform(0, 10_000, features.shape).astype(np.float32)
    elif kind == "impulses":
        features.fill(0)
        features[[0, 1, 14, 15, 30, 96], [1, 4, 6, 12, 50, 175]] = 100
    artifact_path = tmp_path / "model.npz"
    exported = export_tcn(
        model,
        artifact_path,
        **{
            name: frozen.arrays[name]
            for name in ("input_mean", "input_scale", "target_mean", "target_scale")
        },
    )
    loaded = load_tcn(artifact_path)
    for name in exported.arrays:
        np.testing.assert_array_equal(exported.arrays[name], loaded.arrays[name])
    expected = _full_trace(model, frozen, features)
    reloaded = _full_trace(loaded.to_torch(), loaded, features)
    np.testing.assert_array_equal(expected, reloaded)
    window = TCNStream(loaded, "window").predict_chunk(features)
    incremental = TCNStream(loaded, "incremental").predict_chunk(features)
    # The absolute term scales only for the deliberately extreme 10,000-count fixture.
    # Relative tolerance remains 2e-5 and every bin/output is checked, including startup.
    atol = 0.002 if kind == "large" else 0.00002
    np.testing.assert_allclose(window, expected, rtol=2e-5, atol=atol)
    np.testing.assert_allclose(incremental, expected, rtol=2e-5, atol=atol)
    np.testing.assert_allclose(incremental, window, rtol=2e-5, atol=atol)


@pytest.mark.parametrize("mode", ["incremental", "window"])
def test_chunk_partition_trial_continuity_and_file_reset(model_and_artifact, mode):
    model, frozen = model_and_artifact
    features = np.random.default_rng(3).poisson(3, (63, 176))
    single, chunked = TCNStream(frozen, mode), TCNStream(frozen, mode)
    expected = np.stack([single.step(row) for row in features])
    cuts = [0, 1, 7, 7, 15, 23, 41, 62, 63]
    actual = np.concatenate(
        [chunked.predict_chunk(features[start:end]) for start, end in zip(cuts[:-1], cuts[1:])]
    )
    np.testing.assert_array_equal(actual, expected)
    _assert_state_equal(single.state_snapshot(), chunked.state_snapshot())
    # A trial marker at bin 23 does nothing: continuity is achieved by ordinary ingestion.
    np.testing.assert_allclose(actual, _full_trace(model, frozen, features), rtol=2e-5, atol=2e-5)
    chunked.reset()
    np.testing.assert_array_equal(chunked.predict_chunk(features), expected)
    fresh_suffix = TCNStream(frozen, mode).predict_chunk(features[23:])
    assert not np.allclose(fresh_suffix[:14], expected[23:37])
    np.testing.assert_allclose(fresh_suffix[14:], expected[37:], rtol=2e-5, atol=2e-5)


def test_fourteen_bin_burn_in_reproduces_whole_file_training(model_and_artifact):
    model, _ = model_and_artifact
    features = torch.from_numpy(
        np.random.default_rng(6).normal(size=(1, 82, 176)).astype("float32")
    )
    with torch.inference_mode():
        whole = model(features)
        for start, stop in ((0, 17), (7, 31), (14, 35), (29, 58), (67, 82)):
            history_start = max(0, start - 14)
            cropped = model(features[:, history_start:stop])
            torch.testing.assert_close(
                cropped[:, start - history_start :], whole[:, start:stop], rtol=2e-5, atol=2e-5
            )


@pytest.mark.parametrize("mode", ["incremental", "window"])
def test_invalid_input_is_rejected_before_state_changes(model_and_artifact, mode):
    _, frozen = model_and_artifact
    stream = TCNStream(frozen, mode)
    stream.step(np.ones(176))
    before = stream.state_snapshot()
    for value in (np.zeros(175), np.zeros((1, 176)), np.full(176, np.nan), np.full(176, np.inf)):
        with pytest.raises(ValueError):
            stream.step(value)
        _assert_state_equal(before, stream.state_snapshot())
    with pytest.raises(ValueError):
        stream.predict_chunk(np.array([["bad"] * 176]))
    _assert_state_equal(before, stream.state_snapshot())
    with pytest.raises(ValueError):
        stream.predict_chunk(np.vstack([np.ones(176), np.full(176, np.nan)]))
    _assert_state_equal(before, stream.state_snapshot())


def test_state_is_bounded_and_streams_are_independent(model_and_artifact):
    _, frozen = model_and_artifact
    stream = TCNStream(frozen)
    unchanged = TCNStream(frozen)
    empty = unchanged.state_snapshot()
    storage = stream.state_nbytes
    trace = np.random.default_rng(3).poisson(3, (1_300, 176))
    stream.predict_chunk(trace)
    assert stream.state_nbytes == storage
    assert stream.state_snapshot()[0] == 15
    _assert_state_equal(empty, unchanged.state_snapshot())


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nan", "zero_scale", "version", "extra"])
def test_corrupt_artifacts_fail(tmp_path, model_and_artifact, mutation):
    _, frozen = model_and_artifact
    arrays = {name: value.copy() for name, value in frozen.arrays.items()}
    if mutation == "shape":
        arrays["conv1_weight"] = np.zeros((32, 175, 3), dtype="float32")
    elif mutation == "dtype":
        arrays["head_weight"] = arrays["head_weight"].astype("float64")
    elif mutation == "nan":
        arrays["head_bias"][1] = np.nan
    elif mutation == "zero_scale":
        arrays["input_scale"][0] = 0
    elif mutation == "version":
        arrays["format_version"] = np.asarray(99, dtype="int64")
    else:
        arrays["arbitrary"] = np.asarray(0)
    with pytest.raises(ValueError):
        FrozenTCN(arrays)
    path = tmp_path / "corrupt.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_tcn(path)


def test_object_artifact_and_overwrite_are_rejected(tmp_path, model_and_artifact):
    model, frozen = model_and_artifact
    arrays = dict(frozen.arrays)
    arrays["head_bias"] = np.array([object()] * 7, dtype=object)
    path = tmp_path / "object.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_tcn(path)
    path = tmp_path / "existing.npz"
    export_tcn(model, path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        export_tcn(model, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["incremental", "window"])
def test_arithmetic_failure_requires_explicit_reset(model_and_artifact, mode):
    _, frozen = model_and_artifact
    stream = TCNStream(frozen, mode)
    with pytest.raises(FloatingPointError, match="reset"):
        stream.step(np.full(176, np.finfo(np.float64).max))
    with pytest.raises(RuntimeError, match="reset"):
        stream.step(np.zeros(176))
    stream.reset()
    np.testing.assert_array_equal(
        stream.step(np.zeros(176)), TCNStream(frozen, mode).step(np.zeros(176))
    )


def test_inference_reconstruction_does_not_change_torch_rng(model_and_artifact):
    _, frozen = model_and_artifact
    before = torch.get_rng_state().clone()
    frozen.to_torch()
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["incremental", "window"])
def test_nonfinite_layer_result_is_an_explicit_failure(model_and_artifact, mode):
    _, frozen = model_and_artifact
    arrays = {name: value.copy() for name, value in frozen.arrays.items()}
    arrays["head_weight"].fill(np.finfo(np.float32).max)
    # Finite weights can still overflow during execution; structural validity is insufficient.
    stream = TCNStream(FrozenTCN(arrays), mode)
    with pytest.raises(FloatingPointError, match="reset"):
        stream.step(np.zeros(176))
    with pytest.raises(RuntimeError, match="reset"):
        stream.step(np.zeros(176))
