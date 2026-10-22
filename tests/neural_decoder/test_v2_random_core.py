"""Post-hoc control boundaries; small synthetic tests do not test real accuracy."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_experiments.v2 import random_core_control as control
from neural_experiments.v2 import transfer


@pytest.mark.parametrize("channels", [96, 192])
def test_random_core_exactly_matches_scratch_initialization_without_rng_drift(channels):
    plan = control.read(control.PLAN)
    torch.manual_seed(927)
    before = torch.random.get_rng_state().clone()
    actual = control.initial_core(channels, plan, 2)
    assert torch.equal(before, torch.random.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10002)
        scratch = transfer.AdapterTCN(channels, plan["model"]["width"], plan["model"]["dilations"])
    control.assert_core(scratch, actual)
    altered = copy.deepcopy(actual)
    altered["0.bias"][0] += 1
    with pytest.raises(ValueError, match="scratch initialization"):
        control.assert_core(scratch, altered)


def test_exact_adapt_freezes_random_core_but_fits_adapter_and_head():
    rng = np.random.default_rng(10)
    x = rng.poisson(2, (96, 4)).astype(np.float32)
    angle = np.linspace(0, 4, len(x))
    y = np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)
    record = transfer.Recording(
        {"session_id": "synthetic", "participant_id": "synthetic"},
        x,
        y,
        np.ones(len(x), bool),
        [(0, 96)],
    )
    plan = {
        "model": {"width": 4, "dilations": [1], "channel_dropout": 0.1},
        "adaptation": {
            "epochs": 2,
            "chunk_bins": 32,
            "batch_chunks": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
    }
    state = control.initial_core(4, plan, 0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10000)
        initial = transfer.AdapterTCN(4, 4, [1])
    model, stats, history = transfer.adapt(record, plan, 0, "pretrained_frozen_core", state)
    control.assert_core(model, state)
    assert all(not p.requires_grad for p in model.core.parameters())
    assert sorted(name for name, p in model.named_parameters() if p.requires_grad) == [
        "adapter.bias",
        "adapter.weight",
        "head.bias",
        "head.weight",
    ]
    assert not torch.equal(model.adapter.weight, initial.adapter.weight)
    assert not torch.equal(model.head.weight, initial.head.weight)
    assert [row["epoch"] for row in history] == [1, 2]
    for actual, expected in zip(stats, transfer.fit_transform([record]), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_control_plan_binds_original_budget_and_prior_exposure():
    original = control.read(Path(transfer.__file__).with_name("transfer_plan.json"))
    plan = control.read(control.PLAN)
    control.validate_plan(plan, original)
    for key, value in (
        ("budget_seconds", 30),
        ("seeds", [2]),
        ("exploratory", False),
        ("prior_outcome_exposure", ""),
    ):
        changed = copy.deepcopy(plan)
        changed[key] = value
        with pytest.raises(ValueError):
            control.validate_plan(changed, original)
    changed = copy.deepcopy(plan)
    changed["adaptation"]["epochs"] = 20
    with pytest.raises(ValueError, match="adaptation"):
        control.validate_plan(changed, original)


def test_reference_rejects_changed_original_before_loading_any_sample(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"config_path": "missing.json"}))
    plan = control.read(control.PLAN)
    with pytest.raises(ValueError, match="hash mismatch"):
        control.reference(
            plan,
            tmp_path,
            tmp_path / "index.json",
            tmp_path / "public.json",
            tmp_path / "audit.json",
            tmp_path,
        )


def test_null_suffix_still_checks_full_trace_and_scorer_never_uses_transfer_score(monkeypatch):
    monkeypatch.setattr(transfer, "score", lambda *args: pytest.fail("runner scorer was called"))
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    mask = np.array([True, False])
    result = control.scored_metrics(y, y.copy(), mask, "same_session_suffix", 50)
    assert result["r2"] is None and result["n"] == 1
    changed = y.copy()
    changed[1, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite prediction"):
        control.scored_metrics(y, changed, mask, "same_session_suffix", 50)


def test_comparison_weights_people_equally_and_rejects_missing_seed():
    plan = {
        "participants": ["A", "B"],
        "seeds": [0, 1, 2],
        "uncertainty": {"draws": 100, "seed": 1},
    }
    models, learned = [], {"rows": []}
    for person, n_days, value in (("A", 1, -2.0), ("B", 5, -4.0)):
        for seed in plan["seeds"]:
            models.append(
                {
                    "target_participant": person,
                    "seed": seed,
                    "traces": [{"partition": "future_day", "metrics": {"r2": value}}] * n_days,
                }
            )
            learned["rows"].append(
                {
                    "participant": person,
                    "seed": seed,
                    "budget_seconds": 120,
                    "method": "pretrained_frozen_core",
                    "partition": "future_day",
                    "mean_day_r2": value + (1 if person == "A" else 3),
                    "n_days": n_days,
                }
            )
    summary = control.summarize(models, learned, plan)
    contrast = summary["comparisons"][0]
    assert contrast["random_frozen_mean_r2"] == -3
    assert contrast["mean_difference"] == 2
    assert "p_value" not in json.dumps(summary) and "target_gain_met" not in json.dumps(summary)
    with pytest.raises(ValueError, match="three-seed"):
        control.summarize(models[:-1], learned, plan)


def test_public_model_allowlist_omits_unexpected_arrays_and_locations():
    keys = (
        "model_id",
        "target_participant",
        "seed",
        "method",
        "budget_seconds",
        "source_participants",
        "calibration",
        "normalization",
        "model_sha256",
        "initial_core_sha256",
        "training_sha256",
        "training_seconds",
        "parameter_count",
        "trained_parameter_count",
        "trained_parameter_names",
        "reference_learned_model_sha256",
        "reference_learned_core_sha256",
    )
    identity = dict.fromkeys(keys, None)
    identity.update(traces=[], model_path="private/model.npz", input_mean=[1], surprise_array=[123])
    published = control.public_model(identity)
    assert not any(key in published for key in ("model_path", "input_mean", "surprise_array"))
