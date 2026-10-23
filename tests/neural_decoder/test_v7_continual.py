import numpy as np
import pytest

from neural_experiments.v7 import continual, model


def source(n=30000):
    rng = np.random.default_rng(20261027)
    features = {key: rng.random((n, 2)) * 2 for key in ("x_tx3", "x_tx45", "x_sbp")}
    angles = np.arange(n) * 0.01
    return {
        "item": {
            "participant_id": "T2",
            "session_id": "T2_synthetic",
            "post_implant_day": 3,
            "n_channels": 2,
        },
        "features": features,
        "segments": [(0, n)],
        "y": np.column_stack((np.cos(angles), np.sin(angles))),
        "eval_mask": np.ones(n, bool),
    }


def run_case(case, threshold=0.25, directory=None):
    traces, exports = {}, {}

    def save(kind, method, ordinal, payload):
        if kind == "model":
            decoder = payload["decoder"]
            if directory is not None:
                path = directory / f"{method}-{ordinal}.npz"
                decoder.save(path)
                decoder = model.Decoder.load(path)
            exports[method, ordinal] = (decoder, payload["fit"])
            return {"decoder": decoder, "record": {"test_export": ordinal}}
        assert kind == "trace"
        traces[method] = {key: value.copy() for key, value in payload.items()}
        return {"test_trace": True}

    rows, events = continual.evaluate_session(case, threshold, save)
    return rows, events, traces, exports


def test_kl_direction_floor_clipping_and_exclusive_past_geometry():
    raw = np.zeros((14000, 4))
    raw[7500:10500] = np.array([0.1, 0.2, 0.01, 0.02])
    result = continual.drift_evidence(raw, 2, 10500)
    # First30 zero, standard deviations floored to .1/.01, giving normalized
    # mean displacement1/2. Both Gaussian variances are floored to .05.
    assert result["kl_mean"] == pytest.approx(np.mean(np.square([1, 2, 1, 2])) / 0.1)
    assert result["current_start_bin"] == 7500 and result["current_stop_bin"] == 10500
    raw[10500:] = np.nan
    assert continual.drift_evidence(raw, 2, 10500) == result
    identical = np.zeros((10500, 4))
    assert continual.drift_evidence(identical, 2, 10500)["kl_mean"] == 0


@pytest.mark.parametrize("kl,expected", [(0.0, [330, 390, 450, 510]), (10.0, [210, 270, 330, 390])])
def test_trigger_forces_four_probes_without_future_observation(kl, expected):
    selected, end = [], 7500
    for i, second in enumerate(continual.SLOTS_SECONDS):
        decision = continual.choose_probe("triggered240", i, len(selected), end, kl, 0.25)
        if decision["request_probe"]:
            selected.append(second)
            end = second * 50 + continual.PROBE_BINS
    assert selected == expected


def test_threshold_equality_and_periodic_schedule_ignore_kl():
    assert continual.choose_probe("triggered240", 0, 0, 7500, 0.25, 0.25)["request_probe"]
    expected, used, end = [], 0, 7500
    for i, second in enumerate(continual.SLOTS_SECONDS):
        decision = continual.choose_probe("periodic240", i, used, end, 0, 1.0)
        if decision["request_probe"]:
            expected.append(second)
            used += 1
            end = second * 50 + continual.PROBE_BINS
    assert tuple(expected) == continual.PERIODIC_SECONDS


def test_reloaded_models_purchased_masks_zero_interruptions_and_shared_score(tmp_path):
    rows, events, traces, exports = run_case(source(), directory=tmp_path)
    assert len(rows) == 5
    assert len({r["score_mask_sha256"] for r in rows}) == 1
    by_method = {r["method"]: r for r in rows}
    assert all(r["metrics"]["n"] == 22400 for r in rows)
    assert by_method["zero"]["planned_label_seconds"] == 0
    assert by_method["frozen120"]["available_label_seconds"] == 120
    assert by_method["fixed240"]["interruption_seconds"] == 120
    for name in ("periodic240", "triggered240"):
        assert by_method[name]["planned_label_seconds"] == 240
        assert by_method[name]["available_label_seconds"] == 240
        assert by_method[name]["requested_probes"] == 4
        assert by_method[name]["completed_fits"] == 5
        assert by_method[name]["interruption_seconds"] == 128
        assert np.all(traces[name]["prediction"][traces[name]["interruption_mask"]] == 0)
    for event in events:
        if event["event"] != "model_fitted":
            continue
        receipt = event["fit"]
        paid = traces[event["method"]]["purchased_mask"]
        assert receipt["deployment_bin"] == receipt["fit_stop_bin"] + 100
        assert receipt["last_training_bin"] < receipt["fit_stop_bin"]
        assert receipt["training_rows"] == paid[: receipt["fit_stop_bin"]].sum()
        decoder, _ = exports[event["method"], event["fit_ordinal"]]
        np.testing.assert_array_equal(decoder.mean, exports["frozen120", 0][0].mean)
    assert len(events) > 20
    assert len(by_method["periodic240"]["diagnostic_windows"]) == 15


def test_trigger_never_reads_targets_or_evaluation_mask():
    case = source()
    _, events, _, _ = run_case(case)
    case["y"] *= -1
    case["eval_mask"][:] = False
    _, changed, _, _ = run_case(case)
    assert [e for e in events if e["event"] == "probe_decision"] == [
        e for e in changed if e["event"] == "probe_decision"
    ]


def test_unpurchased_future_labels_do_not_change_any_prediction():
    case = source()
    _, _, first, _ = run_case(case)
    case["y"][27000:] *= -100
    _, _, changed, _ = run_case(case)
    for method in continual.METHODS:
        np.testing.assert_array_equal(first[method]["prediction"], changed[method]["prediction"])


def test_future_neural_intervention_cannot_change_past_predictions_or_requests():
    case = source()
    _, events, first, _ = run_case(case)
    for value in case["features"].values():
        value[16600:] *= 100
    _, changed_events, changed, _ = run_case(case)
    for method in continual.METHODS:
        np.testing.assert_array_equal(
            first[method]["prediction"][:16600], changed[method]["prediction"][:16600]
        )
    assert [e for e in events if e["event_bin"] <= 16600] == [
        e for e in changed_events if e["event_bin"] <= 16600
    ]


@pytest.mark.parametrize("n", [1000, 1500, 1501, 7500, 7600, 10500, 11000, 12000, 12100, 12101])
def test_short_recordings_charge_observed_labels_and_do_not_fit_early(n):
    rows, events, traces, _ = run_case(source(n))
    lookup = {r["method"]: r for r in rows}
    assert lookup["frozen120"]["planned_label_seconds"] == 120
    assert lookup["fixed240"]["planned_label_seconds"] == 240
    assert lookup["frozen120"]["available_label_seconds"] == max(0, min(n, 7500) - 1500) * 0.02
    assert lookup["fixed240"]["available_label_seconds"] == max(0, min(n, 13500) - 1500) * 0.02
    for method, trace in traces.items():
        assert len(trace["prediction"]) == n
        assert lookup[method]["available_label_seconds"] == trace["purchased_mask"].sum() * 0.02
    fits = [e for e in events if e["event"] == "model_fitted"]
    assert all(e["event_bin"] < n for e in fits)
    if n <= 7600:
        assert not fits
        assert all(r["metrics"]["r2"] is None for r in rows)
    if n <= 1500:
        assert not events
        assert all(r["unique_requested_label_seconds"] == 0 for r in rows)
        assert all(r["available_label_seconds"] == 0 for r in rows)
    if 10500 < n <= 12100:
        assert lookup["periodic240"]["requested_probes"] == 1
        assert lookup["periodic240"]["completed_fits"] == 1
        assert not np.any(traces["periodic240"]["prediction"][10500:])


def test_acquisition_reset_burn_excluded_from_training_and_scores():
    case = source()
    case["segments"] = [(0, 11000), (11000, 30000)]
    rows, _, traces, exports = run_case(case)
    assert all(r["metrics"]["n"] == 22400 - 62 for r in rows)
    assert not np.any(traces["periodic240"]["eval_mask"][11000:11062])
    assert exports["periodic240", 1][1]["training_rows"] == 7500 - 62


def test_reject_unsupported_threshold_and_noncausal_trigger():
    with pytest.raises(ValueError, match="locked V7B"):
        continual.evaluate_session(source(1000), 0.3)
    with pytest.raises(ValueError, match="complete causal"):
        continual.drift_evidence(np.zeros((1000, 4)), 2, 10500)
    with pytest.raises(ValueError, match="impossible"):
        continual.choose_probe("triggered240", 5, 0, 7500, 0, 0.25)
