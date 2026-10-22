"""Protocol boundaries, independent of real neural data."""

import json

import numpy as np
import pytest

from neural_experiments.v4 import study


def item(sid, day, person="T3"):
    return {"session_id": sid, "post_implant_day": day, "participant_id": person}


def test_histories_exclude_query_same_day_future_other_person_and_layout():
    items = [
        item("old", 0),
        item("a", 1),
        item("b", 2),
        item("bad", 3),
        item("c", 4),
        item("same", 5),
        item("q", 5),
        item("future", 6),
        item("other", 4, "T2"),
    ]
    layouts = {s["session_id"]: (1, 2) for s in items}
    layouts["bad"] = (3, 4)
    assert [s["session_id"] for s in study.histories(items[6], items, layouts)] == ["a", "b", "c"]


def test_insufficient_history_is_not_replaced():
    items = [item(str(i), i) for i in range(5)]
    layouts = {str(i): (1,) for i in range(5)}
    stats = {str(i): {"gram": np.eye(2), "rhs": np.ones((2, 2)), "n": 100} for i in range(4)}
    stats["2"]["n"] = 0
    weights, receipt = study.prior_for(items[-1], items, layouts, stats, 2)
    assert [r["session_id"] for r in receipt] == ["1", "2", "3"]
    assert receipt[1]["disposition"] == "insufficient_labels"
    np.testing.assert_allclose(weights, [[0.5, 0.5], [1, 1]])


def test_future_labels_cannot_change_prefix_fit_statistics():
    rng = np.random.default_rng(4)
    z = np.column_stack((rng.normal(size=(1000, 3)), np.ones(1000)))
    y = rng.normal(size=(1000, 2))
    mask = np.ones(1000, dtype=bool)
    before = study.fitting_stats(z, y, mask, [(0, 700), (700, 1000)], 300)
    y[300:] = np.nan
    mask[300:] = False
    after = study.fitting_stats(z, y, mask, [(0, 700), (700, 1000)], 300)
    assert before["n"] == after["n"] == 238
    np.testing.assert_array_equal(before["rhs"], after["rhs"])


def test_global_selection_excludes_whole_target_person():
    candidates = [{"candidate_id": "g0"}, {"candidate_id": "g1"}]
    rows = [
        {"participant": p, "candidate_id": c, "kind": "global", "metrics": {"r2": s}}
        for p, c, s in [("T2", "g0", 100), ("T2", "g1", -100), ("T3", "g0", 0), ("T3", "g1", 1)]
    ]
    choices = study.choose(rows, candidates, {"participants": ["T2", "T3"]})
    assert [c["candidate_id"] for c in choices] == ["g1", "g0"]


def test_targeted_selection_uses_only_earlier_t3_and_falls_back():
    candidates = [{"candidate_id": "t0"}, {"candidate_id": "t1"}]
    items = [item("early", 1), item("later", 2)]
    rows = [
        {
            "participant": p,
            "session_id": sid,
            "post_implant_day": day,
            "candidate_id": c,
            "kind": "targeted",
            "metrics": {"r2": value},
        }
        for p, sid, day, scores in [
            ("T2", "other", 0, [0, 1]),
            ("T3", "early", 1, [1, 0]),
            ("T3", "later", 2, [-100, 100]),
        ]
        for c, value in zip(["t0", "t1"], scores, strict=True)
    ]
    choices = study.choose_targeted(rows, candidates, items)
    assert choices[0]["candidate_id"] == "t1"
    assert choices[0]["source_session_ids"] == ["other"]
    assert choices[1]["candidate_id"] == "t0"
    assert choices[1]["source_session_ids"] == ["early"]


def test_assessment_requires_bound_source_audit_before_opening_data(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "PASS", "selection_sha256": "wrong"}))
    with pytest.raises(ValueError, match="source audit"):
        study.assess(selection, audit, [tmp_path / "missing"] * 2, tmp_path / "out")
    assert not (tmp_path / "out").exists()
