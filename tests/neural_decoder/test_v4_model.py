import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from neural_experiments.v4 import model


def config(family="tx45", tau=0.2, baseline=None):
    return {
        "family": family,
        "tau_seconds": tau,
        "baseline_tau_seconds": baseline,
        "adapt_gamma": 1.0,
    }


def features(n=260, channels=4):
    rng = np.random.default_rng(41)
    return {
        "x_tx3": rng.poisson(0.8, (n, channels)).astype(float),
        "x_tx45": rng.poisson(0.15, (n, channels)).astype(float),
        "x_sbp": rng.lognormal(1, 0.3, (n, channels)),
    }


def stats(n=100, channels=4, seed=12):
    rng = np.random.default_rng(seed)
    x = np.column_stack((rng.normal(size=(n, channels)), np.ones(n)))
    y = rng.normal(size=(n, 2))
    return model.sufficient(x, y, np.ones(n, dtype=bool)), x, y


@pytest.mark.parametrize("family", ["tx45", "tx3", "tx45_sbp", "tx3_sbp"])
@pytest.mark.parametrize("tau", [0.2, 0.4])
@pytest.mark.parametrize("baseline", [None, 5.0])
def test_full_trace_stream_batch_future_and_reset_parity(family, tau, baseline):
    cfg, f, segments = config(family, tau, baseline), features(), [(0, 151), (151, 260)]
    raw = model.representation(f, segments, cfg)
    _, mean, scale = model.design(raw, 100, 4)
    rng = np.random.default_rng(5)
    decoder = model.Decoder(cfg, mean, scale, rng.normal(0, 0.05, (raw.shape[1] + 1, 2)))
    expected = decoder.predict(f, segments)
    stream = model.StreamingDecoder(decoder)
    actual = []
    for i in range(len(raw)):
        if i in (0, 151):
            stream.reset()
        actual.append(stream.step({k: v[i] for k, v in f.items()}))
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
    changed = {k: v.copy() for k, v in f.items()}
    for value in changed.values():
        value[123:] *= 1000
    np.testing.assert_array_equal(decoder.predict(changed, segments)[:123], expected[:123])
    isolated = {k: v[151:] for k, v in f.items()}
    np.testing.assert_allclose(decoder.predict(isolated, [(0, 109)]), expected[151:], atol=1e-12)
    assert stream.state_nbytes == 2 * mean.nbytes


def test_past_only_baseline_and_startup_exact_semantics():
    f = {key: np.array([[1.0], [9.0], [4.0]]) for key in model.ARRAY_KEYS}
    cfg = config(baseline=5.0)
    raw = model.representation(f, [(0, 3)], cfg)[:, 0]
    fast, slow = np.exp(-0.02 / 0.2), np.exp(-0.02 / 5.0)
    expected1 = (1 - fast) * (3 - 1)
    previous_slow = slow * 1 + (1 - slow) * 3
    np.testing.assert_allclose(
        raw, [0, expected1, fast * expected1 + (1 - fast) * (2 - previous_slow)], atol=1e-14
    )


def test_normalization_uses_only_declared_unlabelled_prefix_and_floors():
    raw = np.column_stack(
        (np.arange(100) * 0.001, np.ones(100), np.arange(100) * 0.0001, np.ones(100))
    )
    x, mean, scale = model.design(raw, 30, 2)
    changed = raw.copy()
    changed[30:] = 1e6
    xx, mm, ss = model.design(changed, 30, 2)
    np.testing.assert_array_equal(mean, mm)
    np.testing.assert_array_equal(scale, ss)
    np.testing.assert_array_equal(x[:30], xx[:30])
    np.testing.assert_array_equal(scale, [0.1, 0.1, 0.01, 0.01])
    assert np.all(xx[30:, :-1] == 8) and np.all(xx[:, -1] == 1)


def test_training_mask_burn_prefix_and_unselected_label_isolation():
    mask = np.ones(260, dtype=bool)
    mask[80] = False
    original = mask.copy()
    selected = model.training_mask(mask, [(0, 151), (151, 260)], stop=240)
    expected = mask[:240].copy()
    expected[:62] = False
    expected[151:213] = False
    np.testing.assert_array_equal(selected, expected)
    np.testing.assert_array_equal(mask, original)
    assert model.training_mask(mask, [(0, 260)], stop=0).size == 0
    _, x, y = stats(240)
    before = model.sufficient(x, y, selected)
    y[~selected] = np.nan
    after = model.sufficient(x, y, selected)
    np.testing.assert_array_equal(before["rhs"], after["rhs"])
    np.testing.assert_array_equal(before["gram"], after["gram"])


def test_history_equals_equal_session_weighted_ridge_not_sample_pooling():
    a, xa, ya = stats(80, seed=4)
    b, xb, yb = stats(320, seed=8)
    fitted = model.solve_history([a, b])
    x, y = np.vstack((xa, xb)), np.vstack((ya, yb))
    weights = np.r_[np.full(80, 1 / 160), np.full(320, 1 / 640)]
    reference = Ridge(alpha=1, fit_intercept=True, solver="cholesky").fit(
        x[:, :-1], y, sample_weight=weights
    )
    np.testing.assert_allclose(fitted[:-1], reference.coef_.T, atol=2e-15)
    np.testing.assert_allclose(fitted[-1], reference.intercept_, atol=2e-15)
    doubled = model.sufficient(
        np.repeat(xa, 2, axis=0), np.repeat(ya, 2, axis=0), np.ones(160, dtype=bool)
    )
    np.testing.assert_allclose(model.solve_history([doubled, b]), fitted, atol=2e-15)


@pytest.mark.parametrize("gamma", [1.0, 10.0])
def test_adaptation_matches_independent_augmented_least_squares(gamma):
    s, x, y = stats()
    prior = np.arange(10).reshape(5, 2) / 30
    diagonal = np.r_[np.full(4, gamma), 0.1 * gamma]
    augmented_x = np.vstack((x / np.sqrt(len(x)), np.diag(np.sqrt(diagonal))))
    augmented_y = np.vstack((y / np.sqrt(len(x)), np.sqrt(diagonal)[:, None] * prior))
    expected = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    np.testing.assert_allclose(model.adapt(s, prior, gamma), expected, atol=2e-15)


def test_no_history_and_low_label_fallback_are_exact_and_independent_copies():
    prior = model.solve_history([], dimension=5)
    np.testing.assert_array_equal(prior, np.zeros((5, 2)))
    nonzero = np.arange(10, dtype=float).reshape(5, 2)
    small, _, _ = stats(49)
    for s in (None, small):
        result = model.adapt(s, nonzero, 1)
        np.testing.assert_array_equal(result, nonzero)
        result[0, 0] = -100
        assert nonzero[0, 0] == 0
    with pytest.raises(ValueError, match="at least50"):
        model.solve_history([small])
    with pytest.raises(ValueError, match="dimension"):
        model.solve_history([])


@pytest.mark.parametrize("penalty", [1.0, 10.0])
@pytest.mark.parametrize("bias", [0, 1])
def test_daily_constrained_and_free_intercept_match_sklearn(penalty, bias):
    s, x, y = stats()
    y += [3, -2]
    s = model.sufficient(x, y, np.ones(len(y), dtype=bool))
    actual = model.solve_daily(s, penalty, bias)
    ref = Ridge(alpha=len(y) * penalty, fit_intercept=bool(bias), solver="cholesky").fit(
        x[:, :-1], y
    )
    np.testing.assert_allclose(actual[:-1], ref.coef_.T, atol=2e-15)
    np.testing.assert_allclose(actual[-1], ref.intercept_ if bias else [0, 0], atol=2e-15)


def test_export_reload_and_feature_only_input_validation(tmp_path):
    cfg, f = config("tx3_sbp", baseline=5.0), features()
    raw = model.representation(f, [(0, 260)], cfg)
    x, mean, scale = model.design(raw, 100, 4)
    s = model.sufficient(x, np.ones((260, 2)), np.ones(260, dtype=bool))
    decoder = model.Decoder(cfg, mean, scale, model.solve_history([s]))
    path = tmp_path / "model.npz"
    decoder.save(path)
    loaded = model.Decoder.load(path)
    np.testing.assert_array_equal(loaded.predict(f, [(0, 260)]), decoder.predict(f, [(0, 260)]))
    with pytest.raises(ValueError, match="labels"):
        loaded.predict({**f, "eval_mask": np.ones(260)}, [(0, 260)])
    stream = model.StreamingDecoder(loaded)
    row = {k: v[0] for k, v in f.items()}
    stream.step(row)
    before = stream.fast.copy(), stream.slow.copy()
    broken = {k: v.copy() for k, v in row.items()}
    broken["x_tx3"][0] = -1
    with pytest.raises(ValueError):
        stream.step(broken)
    np.testing.assert_array_equal(stream.fast, before[0])
    np.testing.assert_array_equal(stream.slow, before[1])


@pytest.mark.parametrize("mutation", ["version", "config", "nonfinite", "scale", "dtype", "shape"])
def test_export_rejects_tampered_numeric_format(tmp_path, mutation):
    data = {
        "format_version": np.asarray(4, dtype=np.int64),
        "config": np.asarray(json.dumps(config())),
        "mean": np.zeros(3),
        "scale": np.ones(3),
        "weights": np.zeros((4, 2)),
    }
    if mutation == "version":
        data["format_version"] = np.asarray(4.0)
    elif mutation == "config":
        data["config"] = np.asarray(json.dumps({**config(), "hidden": 1}))
    elif mutation == "nonfinite":
        data["weights"][0, 0] = np.nan
    elif mutation == "scale":
        data["scale"][0] = 0
    elif mutation == "dtype":
        data["weights"] = data["weights"].astype(np.float32)
    else:
        data["weights"] = np.zeros((3, 2))
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError):
        model.Decoder.load(path)


def test_reject_invalid_config_design_segments_and_stats():
    for changed in [
        {"adapt_gamma": True},
        {"baseline_tau_seconds": 30.0},
        {"tau_seconds": 0.1},
        {"family": "sbp"},
    ]:
        with pytest.raises(ValueError):
            model.validate_config({**config(), **changed})
    with pytest.raises(ValueError):
        model.representation(features(), [(0, 130), (131, 260)], config())
    with pytest.raises(ValueError):
        model.design(np.ones((100, 4)), 0, 4)
    with pytest.raises(ValueError):
        model.training_mask(np.ones(100), [(0, 100)])
    s, _, _ = stats()
    s["rhs"][0, 0] = np.inf
    with pytest.raises(ValueError):
        model.adapt(s, np.zeros((5, 2)), 1)
