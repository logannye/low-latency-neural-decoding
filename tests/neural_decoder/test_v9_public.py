"""Public privacy schemas and exact source-selection guards before fresh access."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from neural_experiments.v3.study import digest, write_json
from neural_experiments.v9 import assessment, prepare_fresh, public


def manifest_fixture():
    config = {
        "representation": "boxcar300",
        "policy": "quota24",
        "training": "trial",
        "target": "window_mean",
        "weighting": "sector",
        "rt_seconds": 0.2,
        "ridge": 1.0,
        "control": "neural",
    }
    row = {
        "participant": "T2",
        "session_id": "T2_day_100_decoding",
        "post_implant_day": 100,
        "method": "quota_sector_mean",
        "config": config,
        "metrics": {"r2": None, "n": 0},
        "secondary": {"window_mean": {"r2": None, "trials": 0}, "trials": 0},
        "calibration": {
            "quota_attained": False,
            "eligible_trials": 2,
            "sector_counts": [1, 0, 1, 0],
            "sector_deficits": [3, 4, 3, 4],
        },
        "score_mask_sha256": "b" * 64,
        "n_bins": 5000,
        "valid_rows": 0,
        "sector_trial_counts": [1, 0, 1, 0],
        "sector_deficits": [3, 4, 3, 4],
        "training": {
            "fallback_zero": True,
            "training_rows": 0,
            "training_trials": 0,
            "private_trials": [{"trial_id": 23, "target": [0.1, 0.2]}],
            "weights": [[1.0, 2.0]],
            "model_path": "/Users/private/model.npz",
        },
        "trace": {"trace_path": "/Users/private/trace.npz"},
        "prediction": [0.1, 0.2],
        "model": {"model_path": "/Users/private/model.npz"},
    }
    selection = {
        "schema_version": 1,
        "study": "V9",
        "source_audit_status": "PENDING",
        "plan_sha256": "1" * 64,
        "index_sha256": "2" * 64,
        "scientific_code_sha256": {},
        "code_sha256": {},
        "primary_method": "quota_sector_mean",
        "configurations": {"quota_sector_mean": config},
        "recipe_choice": {"method": "grid_00", "config": config},
        "grid_selection": {
            "selected_method": "grid_00",
            "ranking": [],
            "no_scorable_sessions_fallback": True,
            "tie_break": "declared_order",
        },
    }
    return {
        "schema_version": 1,
        "study": "V9",
        "status": "complete",
        "phase": "development",
        "plan": {"sha256": "1" * 64, "path": "/Users/private/plan.json"},
        "index": {"sha256": "2" * 64},
        "code_sha256": {},
        "rows": [row],
        "grid_rows": [copy.deepcopy(row)],
        "selection": selection,
        "primary_method": "quota_sector_mean",
        "configurations": {"quota_sector_mean": config},
        "summary": {"selected_sessions": 1, "gates": {"useful_T2_recovery": False}},
        "sessions": [{"artifact_dir": "/Users/private/session", "secondary_trials": [1, 2]}],
    }


def test_aggregate_excludes_private_top_level_and_training_fields_preserves_failures():
    result = public.aggregate(manifest_fixture(), {"status": "PASS", "phase": "development"})
    text = json.dumps(result)
    for marker in ("/Users/", "private_trials", 'prediction"', "model_path", "secondary_trials"):
        assert marker not in text
    assert result["rows"][0]["metrics"]["r2"] is None
    assert result["rows"][0]["training"]["fallback_zero"] is True
    assert result["rows"][0]["sector_deficits"] == [3, 4, 3, 4]
    assert result["summary"]["gates"]["useful_T2_recovery"] is False


@pytest.mark.parametrize("container", ["config", "metrics", "secondary", "calibration"])
def test_nested_sample_bearing_field_rejected(container):
    manifest = manifest_fixture()
    manifest["rows"][0][container]["training_trial_ids"] = [1, 2, 3]
    with pytest.raises(ValueError, match="Unknown field"):
        public.aggregate(manifest, {"status": "PASS", "phase": "development"})


@pytest.mark.parametrize("container", ["summary", "selection"])
def test_nested_manifest_fields_are_not_blindly_copied(container):
    manifest = manifest_fixture()
    manifest[container]["raw_features"] = [[1, 2]]
    with pytest.raises(ValueError, match="Unknown field"):
        public.aggregate(manifest, {"status": "PASS", "phase": "development"})


@pytest.mark.parametrize(
    "private",
    [
        "/Users/private/sample",
        "https://host/?Signature=secret",
        "/private/tmp/sample",
        "/tmp/sample",
        "file:///tmp/sample",
        "Credential=secret",
    ],
)
def test_private_strings_inside_allowed_fields_rejected(private):
    manifest = manifest_fixture()
    manifest["rows"][0]["session_id"] = private
    with pytest.raises(ValueError, match="Private path"):
        public.aggregate(manifest, {"status": "PASS", "phase": "development"})


def test_sample_arrays_cannot_hide_inside_scalar_or_aggregate_vector():
    manifest = manifest_fixture()
    manifest["rows"][0]["metrics"]["r2"] = [0.1, 0.2]
    with pytest.raises(ValueError, match="sample arrays"):
        public.aggregate(manifest, {"status": "PASS", "phase": "development"})
    manifest["rows"][0]["metrics"]["r2"] = 0.1
    manifest["rows"][0]["metrics"]["sse"] = list(range(50))
    with pytest.raises(ValueError, match="shape"):
        public.aggregate(manifest, {"status": "PASS", "phase": "development"})


@pytest.mark.parametrize("status,phase", [("FAIL", "development"), ("PASS", "assessment")])
def test_failed_or_wrong_phase_audit_prevents_aggregation(status, phase):
    with pytest.raises(ValueError, match="audit must pass"):
        public.aggregate(manifest_fixture(), {"status": status, "phase": phase})


@pytest.fixture
def publication_files(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "__file__", str(tmp_path / "neural_experiments/v9/public.py"))
    manifest, audit = tmp_path / "manifest.json", tmp_path / "audit.json"
    value = manifest_fixture()
    write_json(manifest, value)
    write_json(
        audit,
        {
            "status": "PASS",
            "phase": "development",
            "manifest_sha256": digest(manifest),
            "selection_sha256": public.payload_sha256(value["selection"]),
        },
    )
    return manifest, audit, tmp_path / "public.json", tmp_path / "selection.json"


def test_publish_binds_exact_manifest_and_audit(publication_files):
    manifest, audit, output, _ = publication_files
    result = public.publish(manifest, audit, output)
    assert result["private_manifest_sha256"] == digest(manifest)
    assert result["audit_sha256"] == digest(audit)
    assert output.exists()


def test_changed_manifest_rejected_before_any_output(publication_files):
    manifest, audit, output, selection = publication_files
    manifest.write_text(manifest.read_text() + " ")
    for function, destination in ((public.publish, output), (public.seal_selection, selection)):
        with pytest.raises(ValueError, match="exact manifest"):
            function(manifest, audit, destination)
        assert not destination.exists()


def test_sealed_source_choice_binds_audited_payload_without_private_manifest_path(
    publication_files,
):
    manifest, audit, _, output = publication_files
    result = public.seal_selection(manifest, audit, output)
    assert result["source_audit_status"] == "PASS"
    assert result["source_audit_path"] == "audit.json"
    assert result["source_manifest_sha256"] == digest(manifest)
    assert "source_manifest_path" not in result
    assert str(manifest.parent) not in output.read_text()


def test_changed_audited_recipe_digest_prevents_selection_seal(publication_files):
    manifest, audit, _, output = publication_files
    evidence = json.loads(audit.read_text())
    evidence["selection_sha256"] = "f" * 64
    write_json(audit, evidence)
    with pytest.raises(ValueError, match="selected recipe"):
        public.seal_selection(manifest, audit, output)
    assert not output.exists()


@pytest.fixture
def source_guard(tmp_path, monkeypatch):
    source = tmp_path / "neural_experiments/v9/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic source identity\n")
    plan = source.with_name("study-plan.json")
    plan.write_text("{}\n")
    identities = {"neural_experiments/v9/model.py": digest(source)}
    original = manifest_fixture()["selection"]
    original.update(
        plan_sha256=digest(plan), scientific_code_sha256=identities, code_sha256=identities
    )
    audit = tmp_path / "audit.json"
    write_json(
        audit,
        {
            "phase": "development",
            "status": "PASS",
            "manifest_sha256": "a" * 64,
            "selection_sha256": public.payload_sha256(original),
        },
    )
    selected = {
        **original,
        "source_audit_status": "PASS",
        "source_manifest_sha256": "a" * 64,
        "source_audit_sha256": digest(audit),
        "source_audit_path": "audit.json",
    }
    selection = tmp_path / "selection.json"
    write_json(selection, selected)
    checked = []

    def committed(path, expected):
        checked.append(Path(path))
        if digest(path) != expected:
            raise ValueError("committed identity changed")
        return {"path": str(path), "sha256": expected, "git_revision": "synthetic"}

    monkeypatch.setattr(prepare_fresh, "ROOT", tmp_path)
    monkeypatch.setattr(prepare_fresh, "__file__", str(source.with_name("prepare_fresh.py")))
    monkeypatch.setattr(prepare_fresh, "_study_lock", committed)
    monkeypatch.setattr(assessment, "source_hashes", lambda: identities)
    return selection, audit, selected, checked


def test_guard_accepts_exact_committed_source_recipe_and_all_source_hashes(source_guard):
    selection, audit, selected, checked = source_guard
    assert prepare_fresh.validate_selection(selection, digest(selection)) == selected
    assert audit in checked
    assert any(path.name == "model.py" for path in checked)


@pytest.mark.parametrize("change", ["recipe", "unknown", "source_hash", "manifest_hash"])
def test_guard_rejects_any_payload_change_even_if_selection_bytes_are_committed(
    source_guard, change
):
    selection, _, selected, _ = source_guard
    if change == "recipe":
        selected["recipe_choice"]["config"]["ridge"] = 0.1
    elif change == "unknown":
        selected["extra_override"] = True
    elif change == "source_hash":
        selected["scientific_code_sha256"]["neural_experiments/v9/model.py"] = "f" * 64
    else:
        selected["source_manifest_sha256"] = "b" * 64
    write_json(selection, selected)
    with pytest.raises(ValueError, match="audit"):
        prepare_fresh.validate_selection(selection, digest(selection))


@pytest.mark.parametrize("change", ["phase", "status", "manifest", "selection_digest"])
def test_guard_rejects_wrong_source_audit_before_opening_fresh_data(
    source_guard, change, monkeypatch
):
    from neural_experiments.v9 import fresh_data

    selection, audit, selected, _ = source_guard
    evidence = json.loads(audit.read_text())
    field, value = {
        "phase": ("phase", "assessment"),
        "status": ("status", "FAIL"),
        "manifest": ("manifest_sha256", "b" * 64),
        "selection_digest": ("selection_sha256", "b" * 64),
    }[change]
    evidence[field] = value
    write_json(audit, evidence)
    selected["source_audit_sha256"] = digest(audit)
    write_json(selection, selected)
    monkeypatch.setattr(
        fresh_data, "prepare", lambda *_args, **_kwargs: pytest.fail("fresh opened")
    )
    with pytest.raises(ValueError, match="audit"):
        prepare_fresh.run(selection.parent / "fresh", selection, digest(selection))


@pytest.mark.parametrize("location", ["../outside.json", "/tmp/outside.json"])
def test_guard_rejects_audit_paths_outside_repository(source_guard, location):
    selection, _, selected, _ = source_guard
    selected["source_audit_path"] = location
    write_json(selection, selected)
    with pytest.raises(ValueError, match="contained"):
        prepare_fresh.validate_selection(selection, digest(selection))


def test_guard_detects_scientific_code_drift_before_fresh_open(source_guard, monkeypatch):
    selection, _, _, _ = source_guard
    monkeypatch.setattr(assessment, "source_hashes", lambda: {})
    with pytest.raises(ValueError, match="scientific source changed"):
        prepare_fresh.validate_selection(selection, digest(selection))


def test_assessment_rejects_wrong_selection_identity_before_index_access(source_guard, monkeypatch):
    selection, _, _, _ = source_guard
    monkeypatch.setattr(
        assessment.data, "load_index", lambda *_args, **_kwargs: pytest.fail("fresh index accessed")
    )
    with pytest.raises(ValueError, match="exact source selection"):
        assessment.run(
            selection.parent / "index.json", selection.parent / "out", selection, "f" * 64
        )


def test_source_final_rows_separated_from_grid_with_selection_provenance():
    manifest = manifest_fixture()
    final = copy.deepcopy(manifest["rows"][0])
    grid = copy.deepcopy(final)
    grid["method"] = "grid_00"
    manifest["rows"] = [grid, final]
    manifest["final_rows"] = [final]
    manifest["grid_rows"] = [grid]
    manifest["grid_configurations"] = {"grid_00": final["config"]}
    manifest["grid_selection"] = manifest["selection"]["grid_selection"]
    result = public.aggregate(manifest, {"status": "PASS", "phase": "development"})
    assert [row["method"] for row in result["rows"]] == ["quota_sector_mean"]
    assert [row["method"] for row in result["grid_rows"]] == ["grid_00"]
    assert result["grid_configurations"] == manifest["grid_configurations"]
    assert result["grid_selection"] == manifest["grid_selection"]
