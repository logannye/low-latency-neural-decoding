import json

import numpy as np
import pytest
from scipy.signal import lfilter
from sklearn.linear_model import Ridge

from neural_experiments.v3.model import (
    Decoder,
    StreamingDecoder,
    exponential,
    fit,
    prepare_system,
    project_direction,
    representation,
    solve_system,
    training_mask,
)


def config(family="tx", tau=0.2, baseline=None, penalty=1.0):
    return {
        "kind": "regularized",
        "family": family,
        "tau_seconds": tau,
        "baseline_tau_seconds": baseline,
        "ridge_lambda": penalty,
    }


def sample():
    rng = np.random.default_rng(43)
    tx = rng.poisson(0.8, (520, 5)).astype(float)
    sbp = np.exp(rng.normal(size=tx.shape))
    angle = np.linspace(-3, 5, len(tx))
    y = np.column_stack((np.cos(angle), np.sin(angle)))
    mask = rng.random(len(tx)) > 0.2
    segments = [(0, 233), (233, len(tx))]
    return tx, sbp, y, mask, segments


@pytest.mark.parametrize("family", ["tx", "tx_sbp"])
@pytest.mark.parametrize("tau", [0.1, 0.2, 0.4])
@pytest.mark.parametrize("baseline", [None, 30.0])
def test_full_trace_batch_stream_parity_across_resets(family, tau, baseline):
    tx, sbp, y, mask, segments = sample()
    model, _ = fit(tx, sbp, y, mask, segments, config(family, tau, baseline))
    expected = model.predict(tx, sbp, segments)
    stream = StreamingDecoder(model)
    rows = []
    for i in range(len(tx)):
        if i in (0, 233):
            stream.reset()
        rows.append(stream.step(tx[i], sbp[i]))
    np.testing.assert_allclose(rows, expected, rtol=2e-12, atol=2e-12)
    assert np.max(np.linalg.norm(rows, axis=1)) <= 1 + 1e-15
    stream.reset()
    np.testing.assert_allclose(stream.step(tx[0], sbp[0]), rows[0], atol=1e-14)


def test_future_perturbation_and_block_reset_are_causal():
    tx, sbp, y, mask, segments = sample()
    model, _ = fit(
        tx[:150], sbp[:150], y[:150], mask[:150], [(0, 150)], config("tx_sbp", baseline=30.0)
    )
    original = model.predict(tx, sbp, segments)
    changed_tx, changed_sbp = tx.copy(), sbp.copy()
    changed_tx[171:] *= 40
    changed_sbp[171:] += 100
    changed = model.predict(changed_tx, changed_sbp, segments)
    np.testing.assert_array_equal(original[:171], changed[:171])
    changed_tx[:233] = 999
    changed_sbp[:233] = 888
    changed_tx[233:], changed_sbp[233:] = tx[233:], sbp[233:]
    changed = model.predict(changed_tx, changed_sbp, segments)
    np.testing.assert_array_equal(original[233:], changed[233:])


def test_filter_initializes_from_first_observation_and_has_unbounded_causal_support():
    x = np.array([[3.0], [0.0], [0.0], [7.0], [0.0]])
    tau, decay = 0.2, np.exp(-0.02 / 0.2)
    actual = exponential(x, [(0, 3), (3, 5)], tau)
    np.testing.assert_allclose(actual[:, 0], [3, 3 * decay, 3 * decay**2, 7, 7 * decay])
    assert actual[2, 0] > 0


def test_training_mask_excludes_exact_burn_per_block_and_copies():
    mask = np.ones(160, dtype=bool)
    mask[70] = False
    before = mask.copy()
    actual = training_mask(mask, [(0, 90), (90, 160)])
    expected = mask.copy()
    expected[:62] = False
    expected[90:152] = False
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mask, before)
    np.testing.assert_array_equal(training_mask(mask[:30], [(0, 30)]), False)
    with pytest.raises(ValueError, match="boolean"):
        training_mask(np.ones(10), [(0, 10)])
    with pytest.raises(ValueError, match="burn"):
        training_mask(mask, [(0, 160)], burn=-1)


@pytest.mark.parametrize("family", ["tx", "tx_sbp"])
def test_mean_loss_ridge_and_cached_grid_match_sklearn(family):
    tx, sbp, y, mask, segments = sample()
    cfg = config(family, baseline=30.0, penalty=0.1)
    model, receipt = fit(tx, sbp, y, mask, segments, cfg)
    raw = representation(tx, sbp, segments, cfg)
    usable = training_mask(mask, segments)
    x = np.clip((raw - model.mean) / model.scale, -8, 8)
    reference = Ridge(alpha=usable.sum() * cfg["ridge_lambda"], solver="svd").fit(
        x[usable], y[usable]
    )
    np.testing.assert_allclose(model.coefficients, reference.coef_.T, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(model.intercept, reference.intercept_, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        model.predict(tx, sbp, segments),
        project_direction(reference.predict(x)),
        rtol=2e-12,
        atol=2e-12,
    )
    assert receipt["sum_loss_ridge_penalty"] == usable.sum() * 0.1
    system = prepare_system(raw, y, mask, segments, tx.shape[1])
    original_gram = system["gram"].copy()
    solve_system(system, config(family, baseline=30.0, penalty=10.0))
    repeat, _ = solve_system(system, cfg)
    np.testing.assert_array_equal(system["gram"], original_gram)
    np.testing.assert_array_equal(repeat.coefficients, model.coefficients)
    # Unscored labels do not enter the equations or intercept.
    altered = y.copy()
    altered[~usable] = np.nan
    other, _ = fit(tx, sbp, altered, mask, segments, cfg)
    np.testing.assert_array_equal(other.coefficients, model.coefficients)


def test_v2_reference_recipe_matches_float64_sklearn_sum_loss_ridge():
    tx, sbp, y, mask, segments = sample()
    model, receipt = fit(tx, sbp, y, mask, segments, {"kind": "v2_ridge", "family": "tx"})
    raw = (np.sqrt(tx) - model.mean) / model.scale
    weights = np.exp(-np.arange(12) / 12)
    weights /= weights.sum()
    x = np.empty_like(raw)
    for a, b in segments:
        x[a:b] = lfilter(weights, [1.0], raw[a:b], axis=0)
    usable = training_mask(mask, segments)
    ref = Ridge(alpha=10.0, solver="svd").fit(x[usable], y[usable])
    np.testing.assert_allclose(
        model.predict(tx, sbp, segments), ref.predict(x), rtol=2e-12, atol=2e-12
    )
    assert receipt["sum_loss_ridge_penalty"] == 10
    with pytest.raises(ValueError, match="regularized"):
        StreamingDecoder(model)


def test_export_reload_and_invalid_saved_artifacts(tmp_path):
    tx, sbp, y, mask, segments = sample()
    model, _ = fit(tx, sbp, y, mask, segments, config("tx_sbp"))
    path = tmp_path / "model.npz"
    model.save(path)
    restored = Decoder.load(path)
    np.testing.assert_array_equal(
        model.predict(tx, sbp, segments), restored.predict(tx, sbp, segments)
    )
    with pytest.raises(FileExistsError):
        model.save(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    changes = [
        {"format_version": np.array(3.0)},
        {"config": np.asarray(json.dumps({**config("tx_sbp"), "unknown": 1}))},
        {"config": np.asarray(json.dumps({**config("tx_sbp"), "ridge_lambda": True}))},
        {"config": np.asarray(json.dumps({"kind": "v2_ridge", "family": "tx_sbp"}))},
        {"config": np.asarray("[]")},
        {"scale": np.zeros_like(model.scale)},
        {"mean": np.full_like(model.mean, np.nan)},
        {"coefficients": np.zeros((len(model.mean), 3))},
    ]
    for i, change in enumerate(changes):
        invalid = tmp_path / f"bad-{i}.npz"
        np.savez(invalid, **{**arrays, **change})
        with pytest.raises(ValueError):
            Decoder.load(invalid)


@pytest.mark.parametrize("invalid", ["negative", "nan", "shape"])
def test_invalid_observation_rejected_before_stream_state_changes(invalid):
    tx, sbp, y, mask, segments = sample()
    model, _ = fit(tx, sbp, y, mask, segments, config("tx_sbp", baseline=30.0))
    stream = StreamingDecoder(model)
    stream.step(tx[0], sbp[0])
    previous = stream.fast.copy(), stream.slow.copy(), stream.initialized
    bad = sbp[1].copy()
    if invalid == "negative":
        bad[0] = -1
    elif invalid == "nan":
        bad[0] = np.nan
    else:
        bad = bad[:-1]
    with pytest.raises(ValueError):
        stream.step(tx[1], bad)
    np.testing.assert_array_equal(previous[0], stream.fast)
    np.testing.assert_array_equal(previous[1], stream.slow)
    assert stream.initialized == previous[2]
