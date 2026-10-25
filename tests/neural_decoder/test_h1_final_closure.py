"""Synthetic contracts for the final-model evidence producer; no private inputs."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_decoder.artifacts import sha256
from neural_decoder.metrics import FileEvaluation, score_files
from neural_decoder.temporal import CausalTCN, TCNStream, freeze_tcn
from neural_experiments import h1_final_closure as closure


def test_fixed_protocol_rejects_seed_or_tolerance_selection(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "neural_experiments/configs/h1-final-closure-v1.json").read_text())
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert closure.load_config(path)["timing_seed"] == 0
    for key, value in (
        ("timing_seed", 1),
        ("seeds", [0]),
        ("tolerances", {**closure.TOLERANCES, "atol": 0.1}),
    ):
        changed = copy.deepcopy(config)
        changed[key] = value
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="fixed protocol"):
            closure.load_config(path)
    changed = copy.deepcopy(config)
    changed["models"][1]["sha256"] = "a" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="three predetermined"):
        closure.load_config(path)


def test_identity_and_path_escape_rejected(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"exact")
    record = {"path": "payload", "sha256": sha256(path)}
    assert closure.artifact(tmp_path, record) == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        closure.artifact(tmp_path, record)
    with pytest.raises(ValueError, match="inside the repository"):
        closure.artifact(tmp_path, {"path": "../outside", "sha256": record["sha256"]})


def test_approximation_gate_never_replaces_float32_equivalence():
    check = {"file": {"allclose": False}}
    mild = {"S6": {"r2_variance_weighted": -0.001}}
    assert not closure.quality_gate("C", check, mild)["qualified"]
    assert closure.quality_gate("E", check, mild) == {
        "trace_gate": None,
        "score_gate": True,
        "qualified": True,
    }
    cells = {"S6": {"r2_variance_weighted": 0.1}, "S7": {"r2_variance_weighted": -0.0050001}}
    assert not closure.quality_gate("E", check, cells)["qualified"]
    cells["S7"]["r2_variance_weighted"] = -0.005
    assert closure.quality_gate("E", check, cells)["qualified"]
    with pytest.raises(ValueError, match="incomplete"):
        closure.quality_gate("E", check, {})


def test_unscored_bins_and_file_startup_are_part_of_trace_contract():
    torch.set_num_threads(1)
    torch.manual_seed(27)
    frozen = freeze_tcn(CausalTCN())
    counts = np.random.default_rng(28).poisson(0.2, (43, 176)).astype(np.int64)
    expected = closure.whole_file(frozen, counts)
    for arm in ("A", "B"):
        decoder = closure.make_decoder(arm, frozen, Path("unused"))
        prediction = np.concatenate(
            (decoder.predict_chunk(counts[:9]), decoder.predict_chunk(counts[9:]))
        )
        assert closure.trace_check(prediction, expected)["allclose"]
        decoder.reset()
        np.testing.assert_array_equal(decoder.predict_chunk(counts), prediction)
    # Dropping unscored inputs changes later scored predictions even though no
    # labels were passed to the decoder. The full trace check must reject it.
    dropped = TCNStream(frozen).predict_chunk(counts[6:])
    assert not closure.trace_check(dropped, expected[6:])["allclose"]
    mutated = expected.copy()
    mutated[0, 0] += 0.001
    assert not closure.trace_check(mutated, expected)["allclose"]


def test_scoring_concatenates_files_within_day_and_preserves_negative_outputs():
    mask = np.array([True, False, True])
    first = np.arange(21, dtype=float).reshape(3, 7)
    second = first + 10
    a = FileEvaluation("S6_set_1", "S6", first + 2, first, mask)
    b = FileEvaluation("S6_set_2", "S6", second - 5, second, mask)
    scored = score_files([b, a])
    combined = score_files(
        [
            FileEvaluation(
                "S6_all",
                "S6",
                np.r_[a.predictions, b.predictions],
                np.r_[first, second],
                np.r_[mask, mask],
            )
        ]
    )
    assert scored["r2_mean"] == combined["r2_mean"]
    assert scored["num_bins"] == 6 and scored["num_unscored_bins"] == 2
    bad = score_files([FileEvaluation("S6_set_1", "S6", first + 100, first, mask)])
    deltas = closure.score_deltas(bad, score_files([a]))
    assert min(deltas["S6"]["r2_per_output"]) < 0


def test_failed_attempt_retains_start_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(closure, "source_identity", lambda *_: {})
    config = tmp_path / "bad.json"
    config.write_text('{"schema_version": 9}')
    output = tmp_path / "results/neural-h1/failure"
    with pytest.raises(ValueError, match="fixed protocol"):
        closure.run("inventory", config, output, root=tmp_path)
    manifest = closure.read_json(output / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["statuses"]["implementation_execution"] == "failed"
    assert manifest["statuses"]["independent_verification"] == "not_run"
    assert closure.read_json(output / "start.json")["status"] == "running"
    original = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        closure.run("inventory", config, output, root=tmp_path)
    assert (output / "manifest.json").read_bytes() == original


def test_terminal_report_failure_cannot_be_marked_complete(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("{}")
    output = tmp_path / "results/neural-h1/reporter-failure"
    monkeypatch.setattr(closure, "load_config", lambda _: {"inputs": {}})
    monkeypatch.setattr(closure, "source_identity", lambda *_: {})
    monkeypatch.setattr(closure, "authenticate", lambda *_: {"integrity": True})
    original = closure.write_json

    def fail_summary(path, value):
        if path.name == "summary.json":
            raise OSError("synthetic reporting failure")
        return original(path, value)

    monkeypatch.setattr(closure, "write_json", fail_summary)
    with pytest.raises(OSError, match="reporting failure"):
        closure.run("inventory", config, output, root=tmp_path)
    saved = closure.read_json(output / "manifest.json")
    assert saved["status"] == "failed"
    assert saved["statuses"]["artifact_completion"] == "incomplete"
