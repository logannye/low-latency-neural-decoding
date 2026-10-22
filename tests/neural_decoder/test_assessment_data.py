"""Synthetic freeze/release tests; no external recordings are needed or opened."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from neural_decoder import assessment_data as gate
from neural_decoder.data import FEATURE_KEYS, UPSTREAM_PIN, sha256_file


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.STDOUT, text=True)


def _commit(root, *paths):
    _git(root, "add", *paths)
    _git(
        root,
        "-c",
        "user.name=Synthetic Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "synthetic fixture",
    )
    return _git(root, "rev-parse", "HEAD").strip()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path
    cache = root / "data/neural_h1"
    cache.mkdir(parents=True)
    source = root / "src/neural_decoder/core.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    recipe = root / "recipe.json"
    _json(recipe, {"architecture": "synthetic", "epochs": 2, "adaptation": "frozen"})
    model = root / "model.npz"
    np.savez(model, coefficient=np.arange(7))
    _git(root, "init", "-q")
    revision = _commit(root, "src", "recipe.json")
    assets, rows, mapping = [], [], {}
    for day in range(5, 13):
        provider = "held-in-calib" if day == 5 else "held-out-calib"
        for number in (1, 2):
            stamp = f"192501{day + 1:02d}T1200{number:02d}"
            file_id = f"S{day}_set_{number}"
            mapping[stamp] = file_id
            path = f"sub-HumanPitt-{provider}/sub-HumanPitt-{provider}_ses-{stamp}.nwb"
            raw = cache / "raw" / path
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(f"synthetic source {file_id}".encode())
            digest = sha256_file(raw)
            asset_id = f"synthetic-{file_id}"
            assets.append(
                {
                    "asset_id": asset_id,
                    "path": path,
                    "local_path": f"raw/{path}",
                    "sha256": digest,
                    "size": raw.stat().st_size,
                    "provider_digest": {"dandi:sha2-256": digest},
                }
            )
            rows.append(
                {
                    "asset_id": asset_id,
                    "file_id": file_id,
                    "day": f"S{day}",
                    "provider_split": provider,
                    "raw_sha256": digest,
                    "path": path,
                    "n_bins": 1501 + number + day * 7,
                    "n_channels": 176,
                    "n_outputs": 7,
                    "bin_size_s": 0.02,
                    "channel_ids": list(range(176)),
                }
            )
    _json(cache / "manifest.json", {"dandiset": "000954", "version": "draft", "assets": assets})
    _json(
        cache / "schema_audit.json",
        {"source_manifest_sha256": sha256_file(cache / "manifest.json"), "files": rows},
    )
    _json(cache / "split_manifest.json", {"status": "synthetic sealed"})
    monkeypatch.setattr(gate, "session_mapping", lambda path: mapping)
    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "stage": "early",
        "created_at": "2026-10-22T16:00:00+00:00",
        "evaluation_days": ["S5"],
        "provider_split": "held-in-calib",
        "primary_metrics": gate.PRIMARY_METRICS,
        "source": {
            "git_revision": revision,
            "files_sha256": {
                "src/neural_decoder/core.py": sha256_file(source),
                "recipe.json": sha256_file(recipe),
            },
        },
        "recipe": {"path": "recipe.json", "sha256": sha256_file(recipe)},
        "models": [
            {
                "name": "synthetic",
                "path": "model.npz",
                "sha256": sha256_file(model),
                "training_days": [f"S{i}" for i in range(5)],
                "regime": "frozen_zero_calibration",
            }
        ],
        "data": {
            "upstream_pin": UPSTREAM_PIN,
            "manifest_sha256": sha256_file(cache / "manifest.json"),
            "schema_audit_sha256": sha256_file(cache / "schema_audit.json"),
            "split_manifest_sha256": sha256_file(cache / "split_manifest.json"),
        },
    }
    path = root / "freeze-early.json"
    _json(path, freeze)
    return {
        "root": root,
        "cache": cache,
        "freeze": freeze,
        "path": path,
        "source": source,
        "rows": rows,
        "mapping": mapping,
    }


def _validate(setup, path=None, digest=None):
    path = path or setup["path"]
    return gate.validate_freeze(
        path, digest or sha256_file(path), root=setup["root"], cache=setup["cache"]
    )


def _export(setup):
    return gate.export_assessment(
        setup["path"], sha256_file(setup["path"]), root=setup["root"], cache=setup["cache"]
    )


def _fake_reader(monkeypatch):
    called = []

    def read(raw, row, official):
        called.append(raw)
        n = row["n_bins"]
        return (
            {
                "counts": np.zeros((n, 176), dtype=np.uint32),
                "timestamps": np.arange(n) * 0.02,
                "sequence_id": np.arange(n),
                "channel_ids": np.arange(176),
            },
            {
                "targets": np.zeros((n, 7)),
                "eval_mask": np.ones(n, dtype=bool),
                "trial_change": np.zeros(n, dtype=bool),
            },
            {"count_max": 0, "scored_bins": n, "official_loader_exact_array_parity": True},
        )

    monkeypatch.setattr(gate, "_official_loader", lambda path: None)
    monkeypatch.setattr(gate, "_read_arrays", read)
    return called


def test_valid_gate_and_budget_feasibility_do_not_open_raw_files(setup, monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.suffix in {".nwb", ".npz"} and "raw" in path.parts:
            pytest.fail("Metadata validation opened a raw sample file")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert _validate(setup)["stage"] == "early"
    feasibility = gate.calibration_feasibility(setup["cache"])
    assert len(feasibility["files"]) == 14
    assert feasibility["nominal_label_seconds_per_day"] == [0, 10, 30, 60]
    for row in feasibility["files"]:
        assert row["common_score_interval_bins_half_open"] == [1500, row["n_bins"]]
        assert row["prefix_budget_bins"] == [0, 250, 750, 1500]
    assert not (setup["cache"] / "prepared_assessment").exists()


def test_documentation_only_commit_allowed_but_source_mutation_rejected(setup):
    (setup["root"] / "report.md").write_text("Document the frozen result.\n")
    _commit(setup["root"], "report.md")
    assert _validate(setup)["source"]["git_revision"] == setup["freeze"]["source"]["git_revision"]
    setup["source"].write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="artifact identity changed"):
        _validate(setup)


def test_forged_current_hash_cannot_replace_committed_source_identity(setup):
    setup["source"].write_text("VALUE = 2\n")
    setup["freeze"]["source"]["files_sha256"]["src/neural_decoder/core.py"] = sha256_file(
        setup["source"]
    )
    _json(setup["path"], setup["freeze"])
    with pytest.raises(ValueError, match="differs from committed source"):
        _validate(setup)


@pytest.mark.parametrize("change", ["freeze_bytes", "model", "recipe", "training_days", "metrics"])
def test_changed_freeze_or_frozen_artifacts_cannot_release_arrays(setup, monkeypatch, change):
    called = _fake_reader(monkeypatch)
    digest = sha256_file(setup["path"])
    if change == "freeze_bytes":
        setup["path"].write_text(setup["path"].read_text() + " ")
    elif change in {"model", "recipe"}:
        file = setup["root"] / ("model.npz" if change == "model" else "recipe.json")
        file.write_bytes(file.read_bytes() + b"changed")
    else:
        if change == "training_days":
            setup["freeze"]["models"][0]["training_days"].append("S5")
        else:
            setup["freeze"]["primary_metrics"] = {"decoding": "a different metric"}
        _json(setup["path"], setup["freeze"])
        digest = sha256_file(setup["path"])
    with pytest.raises(ValueError):
        gate.export_assessment(setup["path"], digest, root=setup["root"], cache=setup["cache"])
    assert not called
    assert not (setup["cache"] / "prepared_assessment").exists()


def test_provider_day_relabel_rejected_even_with_rehashed_metadata(setup, monkeypatch):
    called = _fake_reader(monkeypatch)
    schema_path = setup["cache"] / "schema_audit.json"
    schema = json.loads(schema_path.read_text())
    schema["files"][0]["day"] = "S6"
    _json(schema_path, schema)
    setup["freeze"]["data"]["schema_audit_sha256"] = sha256_file(schema_path)
    _json(setup["path"], setup["freeze"])
    with pytest.raises(ValueError, match="Schema role/identity"):
        _export(setup)
    assert not called


def test_export_is_separate_immutable_and_feature_only(setup, monkeypatch):
    called = _fake_reader(monkeypatch)
    index_path = _export(setup)
    digest = sha256_file(setup["path"])
    index, files = gate.load_assessment_files(
        index_path, setup["path"], digest, root=setup["root"], cache=setup["cache"]
    )
    assert len(called) == len(files) == 2
    assert index["stage"] == "early"
    for row, counts, targets, mask in files:
        assert counts.shape == (row["n_bins"], 176)
        assert targets.shape == (row["n_bins"], 7)
        assert mask.all()
        with np.load(index_path.parent / row["features"], allow_pickle=False) as features:
            assert set(features.files) == FEATURE_KEYS
    with pytest.raises(FileExistsError):
        _export(setup)
    assert len(called) == 2


def test_index_unauthorized_asset_rejected_before_sample_byte_reads(setup, monkeypatch):
    _fake_reader(monkeypatch)
    index_path = _export(setup)
    index = json.loads(index_path.read_text())
    index["files"][0]["asset_id"] = "synthetic-S6_set_1"
    _json(index_path, index)
    receipt = index_path.parent / "receipt.json"
    record = json.loads(receipt.read_text())
    record["index_sha256"] = sha256_file(index_path)
    _json(receipt, record)
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.suffix == ".npz" and "prepared_assessment" in path.parts:
            pytest.fail("Sample file opened before assessment role rejection")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="Unauthorized"):
        gate.load_assessment_index(
            index_path,
            setup["path"],
            sha256_file(setup["path"]),
            root=setup["root"],
            cache=setup["cache"],
        )


def test_later_stage_preserves_recipe_and_requires_early_parent(setup):
    later = copy.deepcopy(setup["freeze"])
    later.update(
        {
            "stage": "later",
            "evaluation_days": [f"S{i}" for i in range(6, 13)],
            "provider_split": "held-out-calib",
            "created_at": "2026-10-22T17:00:00+00:00",
            "parent_freeze": {"path": "freeze-early.json", "sha256": sha256_file(setup["path"])},
        }
    )
    later["models"][0]["training_days"] = [f"S{i}" for i in range(6)]
    path = setup["root"] / "freeze-later.json"
    _json(path, later)
    assert _validate(setup, path)["stage"] == "later"
    later["parent_freeze"] = {"path": "freeze-later.json", "sha256": sha256_file(path)}
    _json(path, later)
    with pytest.raises(ValueError):
        _validate(setup, path)


def test_export_failure_retains_failure_and_no_completed_index(setup, monkeypatch):
    monkeypatch.setattr(gate, "_official_loader", lambda path: None)

    def fail(*args):
        raise ValueError("synthetic parity failure")

    monkeypatch.setattr(gate, "_read_arrays", fail)
    with pytest.raises(ValueError, match="parity failure"):
        _export(setup)
    directory = next((setup["cache"] / "prepared_assessment").iterdir())
    assert json.loads((directory / "failure.json").read_text())["status"] == "failed"
    assert not (directory / "index.json").exists()
