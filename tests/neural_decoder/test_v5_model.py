"""V5 label windows, fixed warmup, independent convex fits and numeric deployment."""

import hashlib

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from neural_experiments.v5 import model


def synthetic(n=3600):
    rng = np.random.default_rng(654)
    features = {
        "x_tx3": rng.poisson(0.8, (n, 3)).astype(float),
        "x_tx45": rng.poisson(0.2, (n, 3)).astype(float),
        "x_sbp": rng.lognormal(1, 0.4, (n, 3)),
    }
    targets = rng.normal(size=(n, 2)) + [0.5, -0.3]
    mask = rng.random(n) > 0.1
    return features, targets, mask, [(0, 2800), (2800, n)]


@pytest.mark.parametrize("candidate", model.candidate_grid(), ids=lambda c: c["candidate_id"])
@pytest.mark.parametrize("stop", [2250, 3000])
def test_prospective_stage_matches_independent_solve_and_ignores_other_labels(candidate, stop):
    features, y, mask, segments = synthetic()
    raw = model.representation(features, segments, candidate["config"])
    z, _, _ = model.design(raw, 3)
    prior = np.arange(z.shape[1] * 2, dtype=float).reshape(-1, 2) / 20
    actual, receipt = model.fit_stage(z, y, mask, segments, stop, candidate, prior)
    eligible = mask[:stop].copy()
    eligible[:1500] = False
    if stop > 2800:
        eligible[2800:2862] = False
    xx, yy = z[:stop][eligible], y[:stop][eligible]
    if candidate["head"] == "prior10":
        diagonal = np.r_[np.full(z.shape[1] - 1, 10.0), 1.0]
        augmented_x = np.vstack((xx / np.sqrt(len(xx)), np.diag(np.sqrt(diagonal))))
        augmented_y = np.vstack((yy / np.sqrt(len(xx)), np.sqrt(diagonal)[:, None] * prior))
        expected = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    else:
        free = candidate["head"] == "scratch1"
        ref = Ridge(alpha=len(xx) * (1 if free else 10), fit_intercept=free).fit(xx[:, :-1], yy)
        expected = np.vstack((ref.coef_.T, ref.intercept_ if free else np.zeros(2)))
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=1e-12)
    changed = y.copy()
    changed[:1500] = np.nan
    changed[stop:] = np.nan
    changed[:stop][~eligible] = np.nan
    repeat, same_receipt = model.fit_stage(z, changed, mask, segments, stop, candidate, prior)
    np.testing.assert_array_equal(actual, repeat)
    assert same_receipt == receipt
    assert receipt["label_start_bin"] == 1500 and receipt["label_stop_bin"] == stop
    assert receipt["training_rows"] == eligible.sum()
    assert receipt["training_mask_sha256"] == hashlib.sha256(eligible.tobytes()).hexdigest()


def test_fixed_normalizer_and_full_trace_export_causality(tmp_path):
    f, y, mask, segments = synthetic()
    candidate = model.candidate_grid()[3]
    config = candidate["config"]
    raw = model.representation(f, segments, config)
    z, mean, scale = model.design(raw, 3)
    changed = {key: value.copy() for key, value in f.items()}
    for value in changed.values():
        value[1700:] *= 100
    raw2 = model.representation(changed, segments, config)
    z2, mean2, scale2 = model.design(raw2, 3)
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_array_equal(scale, scale2)
    np.testing.assert_array_equal(z[:1700], z2[:1700])
    weights, _ = model.fit_stage(z, y, mask, segments, 3000, candidate)
    decoder = model.Decoder(config, mean, scale, weights)
    decoder.save(tmp_path / "model.npz")
    loaded = model.Decoder.load(tmp_path / "model.npz")
    expected = decoder.predict(f, segments)
    np.testing.assert_array_equal(loaded.predict(f, segments), expected)
    np.testing.assert_array_equal(loaded.predict(changed, segments)[:1700], expected[:1700])
    stream = model.StreamingDecoder(loaded)
    actual = []
    for i in range(len(y)):
        if i in (0, 2800):
            stream.reset()
        actual.append(stream.step({key: value[i] for key, value in f.items()}))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    cut = {key: value[2800:] for key, value in f.items()}
    np.testing.assert_allclose(loaded.predict(cut, [(0, 800)]), expected[2800:], atol=1e-12)


def test_history_stats_use_all_permitted_observations_and_global_stride():
    f, y, mask, segments = synthetic()
    config = model.candidate_grid()[0]["config"]
    raw = model.representation(f, segments, config)
    result = model.historical_stats(raw, y, mask, segments, 3)
    x = np.clip((raw - raw.mean(0)) / np.maximum(raw.std(0), 0.1), -8, 8)
    x = np.column_stack((x, np.ones(len(x))))
    use = mask.copy()
    use[:62] = False
    use[2800:2862] = False
    use &= np.arange(len(use)) % 5 == 0
    assert use[:1500].any()  # Earlier history is not restricted by target warmup.
    np.testing.assert_allclose(result["gram"], x[use].T @ x[use] / use.sum(), atol=1e-15)
    np.testing.assert_allclose(result["rhs"], x[use].T @ y[use] / use.sum(), atol=1e-15)
    assert result["n"] == use.sum()


@pytest.mark.parametrize("stop", [1500, 2250])
def test_zero_label_and_insufficient_label_fallback_are_exact(stop):
    f, y, _, segments = synthetic()
    for candidate in model.candidate_grid():
        raw = model.representation(f, segments, candidate["config"])
        z, _, _ = model.design(raw, 3)
        prior = np.arange(z.shape[1] * 2, dtype=float).reshape(-1, 2)
        mask = np.zeros(len(y), dtype=bool)
        mask[1500:1549] = True
        result, receipt = model.fit_stage(z, y, mask, segments, stop, candidate, prior)
        expected = prior if candidate["head"] == "prior10" else np.zeros_like(prior)
        np.testing.assert_array_equal(result, expected)
        assert receipt["fallback"]
        result[0, 0] = -99
        assert prior[0, 0] == 0


@pytest.mark.parametrize("stop", [1499, 1700, 13501, True])
def test_unplanned_fit_boundaries_rejected(stop):
    f, y, mask, segments = synthetic()
    candidate = model.candidate_grid()[0]
    z, _, _ = model.design(model.representation(f, segments, candidate["config"]), 3)
    with pytest.raises(ValueError, match="planned"):
        model.fit_stage(z, y, mask, segments, stop, candidate)


def test_candidate_identity_and_insufficient_warmup_rejected():
    candidate = model.candidate_grid()[0]
    f, y, mask, segments = synthetic()
    z, _, _ = model.design(model.representation(f, segments, candidate["config"]), 3)
    candidate["head"] = "scratch1"
    with pytest.raises(ValueError, match="identity"):
        model.fit_stage(z, y, mask, segments, 3000, candidate)
    with pytest.raises(ValueError, match="prefix"):
        model.design(np.ones((1499, 3)), 3)


def test_short_recording_retains_requested_cost_and_exact_observed_labels():
    # 113s recording leaves83s of prospective labels, despite a240s request.
    features, y, mask, segments = synthetic(n=5650)
    candidate = model.candidate_grid()[4]
    raw = model.representation(features, segments, candidate["config"])
    z, mean, scale = model.daily_control_design(raw, 3, 13500)
    np.testing.assert_array_equal(mean, raw.mean(0))
    np.testing.assert_array_equal(scale, np.maximum(raw.std(0), np.r_[np.full(3, 0.1), [0.01] * 3]))
    y[:1500] = np.nan
    weights, receipt = model.fit_stage(
        z, y, mask, segments, 13500, candidate, normalization_bins=len(raw)
    )
    assert receipt["requested_label_stop_bin"] == 13500
    assert receipt["observed_label_stop_bin"] == receipt["label_stop_bin"] == 5650
    assert receipt["label_seconds"] == 240
    assert receipt["available_label_seconds"] == 83
    assert receipt["complete_requested_interval"] is False
    assert receipt["normalization_bins"] == 5650
    assert np.isfinite(weights).all()
    use = mask.copy()
    use[:1500] = False
    use[2800:2862] = False
    assert receipt["training_rows"] == use.sum()
    assert receipt["training_mask_sha256"] == hashlib.sha256(use.tobytes()).hexdigest()
    expected = Ridge(alpha=use.sum(), fit_intercept=True).fit(z[use, :-1], y[use])
    np.testing.assert_allclose(weights[:-1], expected.coef_.T, atol=1e-13)
    np.testing.assert_allclose(weights[-1], expected.intercept_, atol=1e-13)


def test_adaptive_candidate_cannot_claim_later_control_normalization():
    f, y, mask, segments = synthetic()
    candidate = model.candidate_grid()[3]
    z, _, _ = model.design(model.representation(f, segments, candidate["config"]), 3)
    with pytest.raises(ValueError, match="normalization"):
        model.fit_stage(z, y, mask, segments, 3000, candidate, normalization_bins=3000)
