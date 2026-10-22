import inspect

import numpy as np
import pytest
from scipy.signal import lfilter

from neural_decoder.adaptation import AdaptiveRidge
from neural_decoder.features import filter_file
from neural_decoder.models import LinearDecoder, LinearModel
from neural_decoder.run_adaptation import apply_decision, select_mixture_window


def model_fixture():
    return LinearModel(
        mean=np.array([1.0, 2.0]),
        scale=np.array([2.0, 3.0]),
        weights=np.array([0.6, 0.3, 0.1]),
        coefficient=np.array([[1.0, -0.25], [0.5, 0.75]]),
        intercept=np.array([0.2, -0.3]),
        history=1,
    )


def equal_state(first, second):
    for key, value in first.items():
        np.testing.assert_array_equal(value, second[key])


@pytest.mark.parametrize("mode,cap", [("frozen", None), ("ema", None), ("bounded", 0.25)])
def test_matches_independent_delayed_ema_and_frozen_trace(mode, cap):
    rng = np.random.default_rng(10)
    counts = rng.poisson(4, size=(123, 2))
    model = model_fixture()
    decoder = AdaptiveRidge(model, mode=mode, cap_sd=cap, tau_seconds=0.3)
    filtered = filter_file(counts, model.weights)
    previous_ema = lfilter(
        [0, decoder.alpha], [1, -(1 - decoder.alpha)], filtered - model.mean, axis=0
    )
    if mode == "frozen":
        previous_ema *= 0
    if mode == "bounded":
        previous_ema = np.clip(previous_ema, -cap * model.scale, cap * model.scale)
    expected = ((filtered - model.mean - previous_ema) / model.scale) @ model.coefficient
    expected += model.intercept
    actual = decoder.predict_chunk(counts)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    if mode == "frozen":
        np.testing.assert_array_equal(actual, LinearDecoder(model).predict_chunk(counts))


def test_first_prediction_precedes_any_adaptation_update():
    model = model_fixture()
    decoder = AdaptiveRidge(model, mode="ema", tau_seconds=0.1)
    counts = np.array([10, 20])
    prediction = decoder.step(counts)
    np.testing.assert_array_equal(prediction, LinearDecoder(model).step(counts))
    np.testing.assert_array_equal(decoder.last_correction, np.zeros(2))
    expected = decoder.alpha * (model.weights[0] * counts - model.mean)
    np.testing.assert_allclose(decoder.ema_deviation, expected)
    decoder.step(counts)
    np.testing.assert_allclose(decoder.last_correction, expected)


@pytest.mark.parametrize("mode,cap", [("frozen", None), ("ema", None), ("bounded", 0.5)])
def test_future_label_mask_trial_chunk_and_file_boundaries(mode, cap):
    rng = np.random.default_rng(21)
    counts = rng.poisson(3, (100, 2))
    target, mask, trial = rng.normal(size=(100, 2)), np.ones(100, bool), np.zeros(100, bool)
    model = model_fixture()
    first = AdaptiveRidge(model, mode=mode, cap_sd=cap)
    second = AdaptiveRidge(model, mode=mode, cap_sd=cap)
    changed = counts.copy()
    changed[37:] += 50
    np.testing.assert_array_equal(
        first.predict_chunk(counts[:37]), second.predict_chunk(changed[:37])
    )
    equal_state(first.state_snapshot(), second.state_snapshot())
    target[:] = 99
    mask[:] = False
    trial[37:] = True
    whole_decoder = AdaptiveRidge(model, mode=mode, cap_sd=cap)
    whole = whole_decoder.predict_chunk(counts)
    chunked = AdaptiveRidge(model, mode=mode, cap_sd=cap)
    parts = [chunked.predict_chunk(counts[a:b]) for a, b in ((0, 4), (4, 37), (37, 100))]
    np.testing.assert_array_equal(whole, np.concatenate(parts))
    equal_state(whole_decoder.state_snapshot(), chunked.state_snapshot())
    chunked.reset()
    np.testing.assert_array_equal(whole, chunked.predict_chunk(counts))
    assert set(inspect.signature(chunked.step).parameters) == {"counts", "sequence_id"}


def test_bounded_applied_correction_and_fixed_memory():
    model = model_fixture()
    bounded = AdaptiveRidge(model, mode="bounded", cap_sd=0.25, tau_seconds=0.01)
    ordinary = AdaptiveRidge(model, mode="ema", tau_seconds=0.01)
    initial = bounded.state_nbytes
    for _ in range(1000):
        bounded.step(np.array([100, 200]))
        ordinary.step(np.array([100, 200]))
        assert np.all(np.abs(bounded.last_correction) <= 0.25 * model.scale)
        assert bounded.state_nbytes == initial
    np.testing.assert_array_equal(bounded.ema_deviation, ordinary.ema_deviation)
    assert np.any(abs(ordinary.last_correction) > 0.25 * model.scale)


def test_invalid_inputs_and_sequence_ids_fail_without_partial_state_changes():
    decoder = AdaptiveRidge(model_fixture(), mode="ema")
    initial = decoder.state_snapshot()
    for counts in (np.array([-1, 1]), np.array([np.nan, 1]), np.zeros(3)):
        with pytest.raises(ValueError):
            decoder.step(counts)
        equal_state(initial, decoder.state_snapshot())
    with pytest.raises(ValueError):
        decoder.step(np.zeros(2), sequence_id=1)
    equal_state(initial, decoder.state_snapshot())
    with pytest.raises(ValueError):
        AdaptiveRidge(model_fixture(), mode="bounded", cap_sd=-1)


def test_mixture_selector_is_contiguous_signed_and_earliest_tie():
    target = np.zeros((6, 7))
    target[:, 6] = [1, 5, -10, 4, 4, 4]
    assert select_mixture_window(target, 3) == (3, 6, 4.0)
    assert select_mixture_window(np.ones((4, 7)), 3) == (0, 3, 1.0)
    with pytest.raises(ValueError, match="complete"):
        select_mixture_window(target[:2], 3)


def test_decision_requires_all_prespecified_guards_and_keeps_frozen_on_failure():
    config = {
        "caps_training_sd": [0.25, 0.5, 1.0],
        "decision": {
            "real_minimum_mean_r2_gain": 0.005,
            "real_maximum_worst_day_harm": 0.01,
        },
    }
    results = {condition: {} for condition in ("real", "observation_offset", "movement_mixture")}
    results["movement_mixture"]["ema"] = {"comparison": {"worst_day_harm": 0.1}}
    for cap in config["caps_training_sd"]:
        name = f"bounded-{cap:g}"
        results["real"][name] = {"comparison": {"mean_r2_difference": 0.01, "worst_day_harm": 0.02}}
        results["observation_offset"][name] = {"comparison": {"mean_r2_difference": 0.1}}
        results["movement_mixture"][name] = {"comparison": {"worst_day_harm": 0.01}}
    decision = apply_decision(results, config)
    assert decision["selected_method"] == "frozen"
    assert not decision["retain_bounded_correction"]
    results["real"]["bounded-0.25"]["comparison"]["worst_day_harm"] = 0.005
    decision = apply_decision(results, config)
    assert decision["selected_method"] == "bounded-0.25"
    results["movement_mixture"]["ema"]["comparison"]["worst_day_harm"] = 0.0
    assert apply_decision(results, config)["selected_method"] == "frozen"
