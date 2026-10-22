import copy
from collections import defaultdict

import numpy as np
import pytest

from neural_experiments.v3.figures import (
    METHODS,
    checked_group,
    paired_budgets,
    primary_comparison,
    render,
)


def synthetic():
    models = []
    # Unequal sessions per person intentionally test equal-person weighting.
    for person, session, value in [("T2", "a", 0.1), ("T2", "b", 0.3), ("T3", "c", 0.8)]:
        for budget in (120, 300):
            for method in METHODS:
                delta = 0.2 if budget == 300 else 0
                if method == "v2_ridge_daily":
                    delta -= 0.4
                if method == "stale_mapping":
                    delta -= 1
                traces = []
                for partition in ("primary", "common_suffix"):
                    start = 15100 if partition == "common_suffix" else (budget + 2) * 50
                    score = value + delta
                    # A dangerous tempting comparison that figures must ignore.
                    if partition == "primary" and budget == 300:
                        score = 100.0
                    traces.append(
                        {
                            "partition": partition,
                            "start_bin": start,
                            "stop_bin": 30000,
                            "mask_sha256": f"{person}-{session}-{start}",
                            "metrics": {"r2": score, "n": 1000, "angular_error_degrees": 60},
                        }
                    )
                models.append(
                    {
                        "target_participant": person,
                        "session_id": session,
                        "method": method,
                        "budget_seconds": budget,
                        "traces": traces,
                    }
                )
    groups = []
    for method in METHODS:
        for budget in (120, 300):
            for partition in ("primary", "common_suffix"):
                people = defaultdict(list)
                for row in models:
                    if row["method"] == method and row["budget_seconds"] == budget:
                        trace = next(t for t in row["traces"] if t["partition"] == partition)
                        people[row["target_participant"]].append(trace["metrics"]["r2"])
                means = [
                    {"participant": p, "r2": float(np.mean(v)), "n_sessions": len(v)}
                    for p, v in people.items()
                ]
                center = float(np.mean([p["r2"] for p in means]))
                groups.append(
                    {
                        "method": method,
                        "budget_seconds": budget,
                        "partition": partition,
                        "people": means,
                        "mean_r2": center,
                        "r2_bootstrap_95": [center - 0.1, center + 0.1],
                    }
                )
    return {"models": models, "summary": {"groups": groups, "operational_gate": {"passed": False}}}


def test_budget_comparison_uses_identical_common_suffix_and_equal_people():
    data = synthetic()
    result = paired_budgets(data)
    selected = [r for r in result["rows"] if r["method"] == "selected_daily"]
    assert [r["mean_r2"] for r in selected] == pytest.approx([0.5, 0.7])
    assert result["window_seconds"] == [302, 600]
    assert result["n_people"] == 2
    assert result["n_sessions"] == 3
    assert result["excluded_unpaired_sessions"] == 0


def test_budget_comparison_intersects_sessions_not_just_people():
    data = synthetic()
    data["models"] = [
        m
        for m in data["models"]
        if not (
            m["method"] == "selected_daily"
            and m["budget_seconds"] == 300
            and m["session_id"] == "b"
        )
    ]
    result = paired_budgets(data)
    assert result["excluded_unpaired_sessions"] == 1
    assert result["n_sessions"] == 2
    assert all(p["n_sessions"] == 1 for r in result["rows"] for p in r["people"].values())
    selected = [r for r in result["rows"] if r["method"] == "selected_daily"]
    assert [r["mean_r2"] for r in selected] == pytest.approx([0.45, 0.65])


@pytest.mark.parametrize("field,value", [("mask_sha256", "tampered"), ("start_bin", 6100)])
def test_budget_pairing_rejects_different_mask_or_primary_window(field, value):
    data = synthetic()
    row = next(
        m for m in data["models"] if m["method"] == "selected_daily" and m["budget_seconds"] == 300
    )
    next(t for t in row["traces"] if t["partition"] == "common_suffix")[field] = value
    with pytest.raises(ValueError, match="mask|window"):
        paired_budgets(data)


def test_primary_requires_matched_case_inventory_and_preserves_negative_stale_scores():
    data = synthetic()
    result = primary_comparison(data)
    assert result["selected"]["T2"]["r2"] == pytest.approx(0.2)
    assert result["reference"]["T2"]["r2"] == pytest.approx(-0.2)
    assert result["stale"]["T2"]["r2"] == pytest.approx(-0.8)
    data["models"].append(copy.deepcopy(data["models"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        checked_group(data, "zero")


def test_published_group_tamper_is_not_silently_plotted():
    data = synthetic()
    group = next(
        g
        for g in data["summary"]["groups"]
        if g["method"] == "selected_daily"
        and g["budget_seconds"] == 120
        and g["partition"] == "primary"
    )
    group["mean_r2"] += 0.5
    with pytest.raises(ValueError, match="equal weight"):
        primary_comparison(data)


def test_svg_output_is_deterministic_and_scope_labels_visible(tmp_path):
    data = synthetic()
    first, second = tmp_path / "first", tmp_path / "second"
    render(data, first, source="Synthetic aggregate test")
    render(data, second, source="Synthetic aggregate test")
    for name in ("v3-participant-accuracy.svg", "v3-methods-and-budget.svg"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert all(line == line.rstrip() for line in (first / name).read_text().splitlines())
    individual = (first / "v3-participant-accuracy.svg").read_text()
    budget = (first / "v3-methods-and-budget.svg").read_text()
    assert "no fresh labels" in individual
    assert "not a clinical-utility threshold" in individual
    assert "[302, 600)" in budget
    assert "2 people and 3 identical sessions" in budget
