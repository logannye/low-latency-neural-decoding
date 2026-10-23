"""Prospective broker, exact confirmed export and counterfactual isolation."""

from pathlib import Path

import numpy as np
import pytest

from neural_experiments.v6 import study


def fixture(tmp_path, monkeypatch, n=12000):
    angle = np.arange(n) * 0.02
    y = np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)
    mask = np.ones(n, dtype=bool)
    target = tmp_path / "targets.npz"
    np.savez(target, y=y, eval_mask=mask)
    rng = np.random.default_rng(631)
    features = {
        "x_tx3": rng.poisson(1, (n, 3)).astype(float),
        "x_tx45": rng.poisson(0.3, (n, 3)).astype(float),
        "x_sbp": rng.lognormal(1, 0.5, (n, 3)),
    }
    cut = n // 2
    segments = [(0, cut), (cut, n)]
    item = {
        "session_id": "T2_day_100",
        "participant_id": "T2",
        "post_implant_day": 100,
        "n_bins": n,
        "n_channels": 3,
        "target_path": target.name,
        "target_sha256": study.digest(target),
        "segments": [{"start": a, "end": b} for a, b in segments],
    }
    case = {"item": item, "features": features, "segments": segments}
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "relative", lambda p: str(Path(p).resolve().relative_to(tmp_path)))
    monkeypatch.setattr(study, "load_case", lambda *_a, **_k: case)
    output = study.init_output(tmp_path / "run")
    replay = study.CaseReplay(
        tmp_path / "index.json",
        item,
        [],
        {item["session_id"]: (0, 1, 2)},
        {"tx45": {}, "tx45_sbp": {}},
        {},
        output,
    )
    return replay, y, mask


def test_broker_bounds_older_views_after_later_request(tmp_path, monkeypatch):
    replay, y, _ = fixture(tmp_path, monkeypatch)
    full, use = replay.broker.request(180)
    assert len(full) == 10500 and not full[:1500].any() and not use[:1500].any()
    prior, _ = replay.broker.request(60)
    assert len(prior) == 4500
    np.testing.assert_array_equal(prior[1500:], y[1500:4500])
    assert replay.broker.requests == [180]
    with pytest.raises(ValueError, match="Unplanned"):
        replay.broker.request(600)


def test_exact_full_fit_is_frozen_before_confirmation_and_never_refit(tmp_path, monkeypatch):
    replay, _, _ = fixture(tmp_path, monkeypatch)
    evidence = replay.get_stage(120)
    assert evidence["fit"]["label_stop_bin"] == 7500
    assert evidence["fit"]["normalization_stop_bin"] == 7500
    assert evidence["confirmation_start_bin"] == 7600
    assert evidence["confirmation_stop_bin"] == 10500
    assert replay.broker.requests == [120, 180]
    assert len(replay.saved_fits) == 1
    before = replay.models[120].weights.copy()
    decision = {
        "action": "accept",
        "training_seconds": 120,
        "model_sha256": evidence["model_sha256"],
    }
    deployed = replay.decoder_for(decision)
    np.testing.assert_array_equal(before, deployed.weights)
    assert len(replay.saved_fits) == 1
    path = tmp_path / evidence["model_path"]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="changed"):
        replay.decoder_for(decision)


def test_confirmation_and_future_labels_cannot_change_fitted_model(tmp_path, monkeypatch):
    replay, y, mask = fixture(tmp_path, monkeypatch)
    evidence = replay.get_stage(120)
    original = replay.models[120].weights.copy()
    y[7500:] *= -1
    np.savez(replay.broker.path, y=y, eval_mask=mask)
    replay.item["target_sha256"] = study.digest(replay.broker.path)
    replay.stages.clear()
    changed = replay.get_stage(120)
    np.testing.assert_array_equal(original, replay.models[120].weights)
    assert evidence["model_sha256"] == changed["model_sha256"]
    assert evidence["confirmation"] != changed["confirmation"]
    y[10500:] *= -1
    np.savez(replay.broker.path, y=y, eval_mask=mask)
    replay.item["target_sha256"] = study.digest(replay.broker.path)
    replay.stages.clear()
    assert replay.get_stage(120)["confirmation"] == changed["confirmation"]


def test_prior_confirmation_costs_labels_and_has_uniform_purge(tmp_path, monkeypatch):
    replay, _, _ = fixture(tmp_path, monkeypatch)
    evidence = replay.get_stage(0)
    assert evidence["training_seconds"] == 0
    assert evidence["requested_label_seconds"] == 60
    assert evidence["confirmation_start_bin"] == 1600
    assert replay.broker.requests == [60]
    assert len(replay.saved_fits) == 1


def test_future_neural_features_cannot_change_fit_or_confirmation(tmp_path, monkeypatch):
    replay, _, _ = fixture(tmp_path, monkeypatch)
    evidence = replay.get_stage(120)
    for values in replay.case["features"].values():
        values[10500:] = 1e6
    # Recompute the producer's causal caches as if the altered session were loaded fresh.
    for family in replay.raw:
        config = next(c["config"] for c in replay.candidates if c["config"]["family"] == family)
        replay.raw[family] = study.core.representation(
            replay.case["features"], replay.case["segments"], config
        )
    replay.stages.clear()
    changed = replay.get_stage(120)
    assert changed["model_sha256"] == evidence["model_sha256"]
    assert changed["confirmation"] == evidence["confirmation"]


def test_short_checkpoint_keeps_requested_and_available_cost(tmp_path, monkeypatch):
    replay, _, _ = fixture(tmp_path, monkeypatch, n=5000)
    evidence = replay.get_stage(240)
    assert evidence["unavailable"] == "complete_fit_and_confirmation_not_observed"
    assert evidence["requested_label_seconds"] == 300
    assert evidence["available_label_seconds"] == 70
    assert replay.saved_fits == []
    controls = replay.controls()
    assert set(controls) == {"zero", "history0", "daily120", "daily240", "daily360"}
    assert replay.broker.requests == [300]
    assert all(r["fit"]["normalization_stop_bin"] == 5000 for r in replay.saved_fits)


def test_below_warmup_has_zero_controls_and_no_fit(tmp_path, monkeypatch):
    replay, _, _ = fixture(tmp_path, monkeypatch, n=1000)
    assert replay.get_stage(0)["available_label_seconds"] == 0
    assert all(not v.weights.any() for v in replay.controls().values())
    assert replay.saved_fits == []


def test_real_controller_integration_preserves_cost_on_short_recording(tmp_path, monkeypatch):
    replay, _, _ = fixture(tmp_path, monkeypatch)
    history = {
        "current_session_id": replay.item["session_id"],
        "current_post_implant_day": 100,
        "past_learning_curves": [],
    }
    decision = study.policy.decide(history, replay.get_stage, study.policy.policy_grid()[0])
    assert decision["action"] == "abstain"
    assert decision["requested_label_seconds"] == 300
    assert decision["available_label_seconds"] == 210
    assert decision["reason"] == "incomplete_requested_confirmation"
    assert not replay.decoder_for(decision).weights.any()
