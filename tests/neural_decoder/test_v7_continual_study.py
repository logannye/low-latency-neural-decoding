import json
from pathlib import Path

import numpy as np
import pytest

from neural_experiments.v7 import continual_metrics, study_continual


@pytest.fixture
def synthetic_study(monkeypatch, tmp_path):
    """Replace acquisition only; exercise the actual engine, exports and ranking."""
    plan = json.loads(Path(study_continual.__file__).with_name("continual-plan.json").read_text())
    index_path = tmp_path / "index.json"
    index_path.write_text("{}")
    sessions, cases = [], {}
    for participant, n in (("T2", 30000), ("T5", 30000), ("T3", 1000)):
        sid = f"{participant}_synthetic"
        rng = np.random.default_rng(int(participant[1:]))
        item = {
            "participant_id": participant,
            "session_id": sid,
            "post_implant_day": 1,
            "n_channels": 1,
            "n_bins": n,
            "feature_path": f"{sid}.features.npz",
            "target_path": f"{sid}.targets.npz",
            "segments": [{"start": 0, "end": n}],
        }
        features = {key: rng.random((n, 1)) for key in ("x_tx3", "x_tx45", "x_sbp")}
        angles = np.arange(n) * 0.01
        case = {
            "item": item,
            "features": features,
            "segments": [(0, n)],
            "y": np.column_stack((np.cos(angles), np.sin(angles))),
            "eval_mask": np.ones(n, bool),
        }
        np.savez_compressed(tmp_path / item["feature_path"], **features)
        np.savez_compressed(
            tmp_path / item["target_path"], y=case["y"], eval_mask=case["eval_mask"]
        )
        sessions.append(item)
        cases[sid] = case
    index = {"sessions": sessions}
    calls = []

    def load_index(path, *, expected_role, verify_hashes):
        assert Path(path) == index_path and verify_hashes
        calls.append(("index", expected_role))
        return index

    def load_case(path, item, *, labels):
        assert Path(path) == index_path and labels
        calls.append(("case", item["session_id"]))
        return cases[item["session_id"]]

    def load_npz(path):
        calls.append(("npz", Path(path).name))
        with np.load(path, allow_pickle=False) as values:
            return {key: values[key] for key in values.files}

    actual_digest = study_continual.digest

    def pinned_digest(path):
        if Path(path) == index_path:
            return plan["development_index_sha256"]
        return actual_digest(path)

    monkeypatch.setattr(study_continual, "digest", pinned_digest)
    monkeypatch.setattr(study_continual, "relative", lambda path: str(Path(path).resolve()))
    monkeypatch.setattr(study_continual.platform, "platform", lambda: "synthetic-platform")
    monkeypatch.setattr(
        study_continual.subprocess, "check_output", lambda *a, **kw: "synthetic-head\n"
    )
    monkeypatch.setattr(study_continual.data, "load_index", load_index)
    monkeypatch.setattr(study_continual.data, "load_case", load_case)
    monkeypatch.setattr(study_continual.fresh_data, "load_index", load_index)
    monkeypatch.setattr(study_continual.fresh_data, "load_features", load_npz)
    monkeypatch.setattr(study_continual.fresh_data, "load_targets", load_npz)
    return {"index": index_path, "calls": calls, "actual_digest": actual_digest, "cases": cases}


def test_source_grid_exact_exports_unique_threshold_paths_and_selection(synthetic_study, tmp_path):
    output = tmp_path / "source"
    result = study_continual.run("source", synthetic_study["index"], output)
    assert result["status"] == "complete"
    assert result["git_head"] == "synthetic-head"
    assert len(result["rows"]) == 3 * 4 * 5
    assert synthetic_study["calls"].count(("index", "development")) == 1
    assert result["selection"] == continual_metrics.select_source(result["rows"])
    choices = {r["participant"]: r["threshold"] for r in result["selection"]["choices"]}
    selected = [r for r in result["rows"] if r["threshold"] == choices[r["participant"]]]
    assert result["summary"] == continual_metrics.summarize(selected)
    fits = [e for e in result["events"] if e["event"] == "model_fitted"]
    model_paths = [e["model_path"] for e in fits]
    trace_paths = [r["trace_path"] for r in result["rows"]]
    assert len(model_paths) == len(set(model_paths)) == len(list((output / "models").glob("*.npz")))
    assert len(trace_paths) == len(set(trace_paths)) == 60
    assert len(list((output / "traces").glob("*.npz"))) == 60
    for fit in fits:
        assert f"-{fit['threshold']:g}-" in Path(fit["model_path"]).name
        assert synthetic_study["actual_digest"](fit["model_path"]) == fit["model_sha256"]
    for row in result["rows"]:
        assert synthetic_study["actual_digest"](row["trace_path"]) == row["trace_sha256"]
        with np.load(row["trace_path"], allow_pickle=False) as trace:
            assert set(trace.files) == {
                "prediction",
                "eval_mask",
                "interruption_mask",
                "purchased_mask",
            }
            assert trace["prediction"].shape == (row["n_bins"], 2)
            assert not np.any(trace["prediction"][trace["interruption_mask"]])
    for path, expected in result["code_sha256"].items():
        assert synthetic_study["actual_digest"](output / "source_snapshot" / path) == expected
    assert json.loads((output / "manifest.json").read_text()) == result
    with pytest.raises(FileExistsError):
        study_continual.run("source", synthetic_study["index"], output)


def selection_fixture(tmp_path, *, status="PASS", code_hashes=None):
    hashes = code_hashes or {
        path: study_continual.digest(study_continual.ROOT / path)
        for path in study_continual.SOURCES
    }
    selection = {
        "plan_sha256": study_continual.PLAN_SHA,
        "source_audit_status": status,
        "scientific_code_sha256": hashes,
        "selection": {
            "choices": [
                {"participant": "T2", "threshold": 0.25},
                {"participant": "T5", "threshold": 0.5},
                {"participant": "T3", "threshold": 1.0},
            ]
        },
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    return path, study_continual.digest(path)


def test_fresh_selected_threshold_target_dictionary_and_summary_integration(
    synthetic_study, tmp_path
):
    selection, checksum = selection_fixture(tmp_path)
    result = study_continual.run(
        "assessment",
        synthetic_study["index"],
        tmp_path / "assessment",
        selection_path=selection,
        selection_sha=checksum,
    )
    assert result["status"] == "complete" and len(result["rows"]) == 15
    assert result["summary"] == continual_metrics.summarize(result["rows"])
    assert result["source_selection"]["sha256"] == checksum
    expected = {"T2": 0.25, "T5": 0.5, "T3": 1.0}
    assert all(r["threshold"] == expected[r["participant"]] for r in result["rows"])
    assert synthetic_study["calls"].count(("index", "fresh")) == 1
    assert sum(call[0] == "npz" for call in synthetic_study["calls"]) == 6
    assert not any(call[0] == "case" for call in synthetic_study["calls"])


@pytest.mark.parametrize("failure", ["missing", "digest", "audit", "code"])
def test_fresh_does_not_open_feature_or_target_arrays_without_valid_selection(
    synthetic_study, tmp_path, failure
):
    path, checksum = selection_fixture(
        tmp_path,
        status="FAIL" if failure == "audit" else "PASS",
        code_hashes={"changed": "identity"} if failure == "code" else None,
    )
    if failure == "missing":
        path = checksum = None
    if failure == "digest":
        checksum = "0" * 64
    with pytest.raises(ValueError):
        study_continual.run(
            "assessment",
            synthetic_study["index"],
            tmp_path / "rejected",
            selection_path=path,
            selection_sha=checksum,
        )
    assert not (tmp_path / "rejected").exists()
    assert not any(call[0] in ("npz", "case") for call in synthetic_study["calls"])
