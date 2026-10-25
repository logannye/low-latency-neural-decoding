"""Mutation tests for independent final-model evidence admission (synthetic only)."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnx

from neural_decoder.metrics import FileEvaluation, score_files
from neural_decoder.temporal import CausalTCN, TCNStream, export_tcn
from neural_experiments import audit_h1_final_closure as audit
from neural_experiments import h1_final_closure as producer
from neural_experiments.compression import convert


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_prediction(root, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return {"path": str(path.relative_to(root)), "sha256": audit.digest(path)}


def producer_scores(files, predictions):
    return score_files(
        [
            FileEvaluation(
                item["file_id"], item["day"], predictions[item["file_id"]], targets, mask
            )
            for item, _, targets, mask in files
        ]
    )


@pytest.fixture(scope="module")
def synthetic_evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("final-quality-audit")
    run = root / "quality"
    run.mkdir()
    rng = np.random.default_rng(238)
    files = []
    for day, number, bins in (("S6", 1, 7), ("S6", 2, 12), ("S10", 1, 9), ("S10", 2, 13)):
        fid = f"{day}_set_{number}"
        counts = rng.poisson(0.5, (bins, 176)).astype(np.float32)
        targets = rng.normal(size=(bins, 7)).astype(np.float32)
        mask = np.ones(bins, dtype=bool)
        mask[::4] = False
        files.append(({"file_id": fid, "day": day}, counts, targets, mask))
    config = {"models": [], "inputs": {"historical_manifest": {"path": "history/manifest.json"}}}
    inventory, historical, summary = {"models": []}, {"models": []}, {"models": []}
    for seed in range(3):
        torch.manual_seed(90 + seed)
        path = root / f"model-{seed}.npz"
        frozen = export_tcn(CausalTCN(), path)
        record = {
            "name": f"tcn-seed-{seed}",
            "path": path.name,
            "sha256": audit.digest(path),
            "seed": seed,
            "kind": "tcn",
        }
        config["models"].append({key: record[key] for key in ("name", "path", "sha256", "seed")})
        folder = run / f"seed-{seed}"
        folder.mkdir()
        convert(frozen, folder)
        refs = {
            item["file_id"]: producer.whole_file(frozen, counts) for item, counts, _, _ in files
        }
        hist = {
            item["file_id"]: TCNStream(frozen).predict_chunk(counts) for item, counts, _, _ in files
        }
        history_records = {
            fid: save_prediction(root, root / "history" / record["name"] / f"{fid}.npy", pred)
            for fid, pred in hist.items()
        }
        historical_scores = producer_scores(files, hist)
        write_json(root / "history" / record["name"] / "scores.json", historical_scores)
        historical["models"].append(
            {
                "model": record,
                "scores": historical_scores,
                "prediction_sha256": {
                    fid: entry["sha256"] for fid, entry in history_records.items()
                },
            }
        )
        inventory["models"].append(
            {
                "seed": seed,
                "model": record,
                "model_bytes": path.stat().st_size,
                "historical_predictions": history_records,
                "historical_scores": historical_scores,
            }
        )
        reference = {
            "predictions": {
                fid: save_prediction(run, folder / "reference" / f"{fid}.npy", pred)
                for fid, pred in refs.items()
            },
            "scores": producer_scores(files, refs),
            "historical_trace": {fid: producer.trace_check(refs[fid], hist[fid]) for fid in refs},
            "historical_scores_match": True,
        }
        reference["historical_delta_per_day"] = producer.score_deltas(
            reference["scores"], historical_scores
        )
        reference["historical_gate"] = producer.quality_gate(
            "A", reference["historical_trace"], reference["historical_delta_per_day"]
        )
        write_json(folder / "reference.json", reference)
        entry = {
            "seed": seed,
            "model": record,
            "fingerprint": audit.fingerprint(frozen.arrays),
            "parameter_count": frozen.parameter_count,
            "reference": reference,
            "arms": {},
        }
        for arm in audit.ARMS:
            predictions, state_bytes = {}, None
            for item, counts, _, _ in files:
                engine = producer.make_decoder(arm, frozen, folder)
                predictions[item["file_id"]] = engine.predict_chunk(counts)
                state_bytes = engine.state_nbytes
            precision = "dynamic_int8" if arm == "E" else "float32"
            graph = (
                producer._graph_record(
                    folder / producer.GRAPH_NAMES[arm], run, entry["fingerprint"], precision
                )
                if arm in producer.GRAPH_NAMES
                else None
            )
            result = {
                "status": "complete",
                "precision": precision,
                "graph": graph,
                "predictions": {
                    fid: save_prediction(run, folder / arm / f"{fid}.npy", pred)
                    for fid, pred in predictions.items()
                },
                "scores": producer_scores(files, predictions),
                "numeric_state_bytes": state_bytes,
                "trace_checks": {
                    fid: producer.trace_check(predictions[fid], refs[fid]) for fid in refs
                },
            }
            result["delta_per_day"] = producer.score_deltas(result["scores"], reference["scores"])
            result.update(
                producer.quality_gate(arm, result["trace_checks"], result["delta_per_day"])
            )
            entry["arms"][arm] = result
            write_json(folder / f"{arm}-result.json", result)
        summary["models"].append(entry)
    return root, config, summary, inventory, files, historical


def evidence_copy(synthetic_evidence, tmp_path):
    base, config, summary, inventory, files, historical = synthetic_evidence
    root = tmp_path / "evidence"
    shutil.copytree(base, root)
    return root, *copy.deepcopy((config, summary, inventory, files, historical))


def run_model_audit(evidence):
    root, config, summary, inventory, files, historical = evidence
    return audit._audit_models(
        root / "quality", root, config, summary, inventory, files, historical
    )


def test_independent_three_seed_full_trace_and_day_scores(synthetic_evidence, tmp_path):
    rows, eligibility, failures = run_model_audit(evidence_copy(synthetic_evidence, tmp_path))
    assert len(rows) == 3 * 5 * 2
    assert eligibility == dict.fromkeys(audit.ARMS, True)
    assert not failures


@pytest.mark.parametrize("arm", ["A", "B", "C", "D", "E"])
def test_rehashed_plausible_finite_prediction_mutation_rejected(synthetic_evidence, tmp_path, arm):
    evidence = evidence_copy(synthetic_evidence, tmp_path)
    root, _, summary, *_ = evidence
    result = summary["models"][0]["arms"][arm]
    record = result["predictions"]["S6_set_1"]
    path = root / "quality" / record["path"]
    values = np.load(path, allow_pickle=False)
    # First bin is unscored: full-trace verification must still reject it.
    values[0, 3] += np.float32(0.012)
    np.save(path, values, allow_pickle=False)
    record["sha256"] = audit.digest(path)
    write_json(root / "quality" / "seed-0" / f"{arm}-result.json", result)
    with pytest.raises(ValueError, match="independent inference"):
        run_model_audit(evidence)


def test_mask_mutation_rejected_by_independent_historical_scores(synthetic_evidence, tmp_path):
    evidence = evidence_copy(synthetic_evidence, tmp_path)
    files = evidence[4]
    files[0][3][0] = True
    with pytest.raises(ValueError, match="historical scores"):
        run_model_audit(evidence)


def test_changed_model_rejected_before_graph_or_trace_use(synthetic_evidence, tmp_path):
    evidence = evidence_copy(synthetic_evidence, tmp_path)
    root, config, *_ = evidence
    path = root / config["models"][0]["path"]
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    arrays["head_bias"][0] += 0.01
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="changed evidence"):
        run_model_audit(evidence)


def test_rehashed_graph_mutation_cannot_hide_behind_model_metadata(synthetic_evidence, tmp_path):
    evidence = evidence_copy(synthetic_evidence, tmp_path)
    root, _, summary, *_ = evidence
    result = summary["models"][0]["arms"]["D"]
    record = result["graph"]
    path = root / "quality" / record["path"]
    graph = onnx.load(path)
    relu = next(node for node in graph.graph.node if node.op_type == "Relu")
    relu.op_type = "Identity"
    onnx.save(graph, path)
    record["sha256"], record["bytes"] = audit.digest(path), path.stat().st_size
    ops = [node.op_type for node in graph.graph.node]
    record["operations"] = {op: ops.count(op) for op in sorted(set(ops))}
    write_json(root / "quality" / "seed-0" / "D-result.json", result)
    with pytest.raises(ValueError, match="beyond exact transposes"):
        run_model_audit(evidence)


@pytest.mark.parametrize("mutation", ["scale", "weight", "nonlayout"])
def test_quantized_graph_exact_arithmetic_mutations_rejected(synthetic_evidence, mutation):
    root = synthetic_evidence[0]
    rewritten = onnx.load(root / "quality/seed-0/quantization-input.onnx")
    compressed = onnx.load(root / "quality/seed-0/dynamic-int8.onnx")
    if mutation == "nonlayout":
        next(node for node in compressed.graph.node if node.op_type == "Relu").op_type = "Identity"
    else:
        name = "conv1_weight_scale" if mutation == "scale" else "conv1_weight_quantized"
        tensor = next(value for value in compressed.graph.initializer if value.name == name)
        value = onnx.numpy_helper.to_array(tensor).copy()
        value.flat[0] += 0.0001 if mutation == "scale" else 1
        tensor.CopyFrom(onnx.numpy_helper.from_array(value, name))
    with pytest.raises(ValueError, match="quantization differs|non-layout operations"):
        audit.verify_quantization(rewritten, compressed)


def test_gate_checks_every_day_and_keeps_float_parity_strict():
    checks = {"file": {"allclose": True}}
    deltas = {f"S{day}": {"r2_variance_weighted": 0.002} for day in range(6, 13)}
    deltas["S12"]["r2_variance_weighted"] = -0.005001
    assert np.mean([row["r2_variance_weighted"] for row in deltas.values()]) > 0
    assert not audit.decide_gate("E", checks, deltas)["qualified"]
    deltas["S12"]["r2_variance_weighted"] = -0.005
    assert audit.decide_gate("E", checks, deltas)["qualified"]
    assert not audit.decide_gate("D", checks, deltas)["qualified"]
    assert not audit.decide_gate(
        "A", {"file": {"allclose": False}}, {"S6": {"r2_variance_weighted": 0}}
    )["qualified"]


def test_independent_score_groups_files_before_r2_and_retains_negative_outputs():
    files, predictions = [], {}
    for number, offset in ((1, 0), (2, 10)):
        fid = f"S6_set_{number}"
        y = np.tile(np.arange(4, dtype=float)[:, None] + offset, (1, 7))
        p = y.copy()
        p[:, 6] += 100
        files.append(({"file_id": fid, "day": "S6"}, np.zeros((4, 176)), y, np.ones(4, bool)))
        predictions[fid] = p
    scores = audit.independent_scores(files, predictions)
    assert scores["per_day"]["S6"]["num_files"] == 2
    assert scores["per_day"]["S6"]["r2_per_output"][6] < 0
    audit.assert_equal(scores, producer_scores(files, predictions))


@pytest.mark.parametrize("value", ['{"a": 1, "a": 2}', '{"value": NaN}'])
def test_ambiguous_json_rejected(tmp_path, value):
    path = tmp_path / "bad.json"
    path.write_text(value)
    with pytest.raises(ValueError):
        audit.read_json(path)


def test_prediction_artifact_alias_and_escape_rejected(tmp_path):
    path = tmp_path / "prediction.npy"
    np.save(path, np.zeros((2, 7), np.float32))
    record = {"path": path.name, "sha256": audit.digest(path)}
    with pytest.raises(ValueError, match="alias"):
        audit.load_prediction_records(tmp_path, {"a": record, "b": record}, ["a", "b"])
    with pytest.raises(ValueError, match="contained"):
        audit.bound_path(tmp_path, "../prediction.npy")


@pytest.fixture
def receipt_evidence(tmp_path, monkeypatch):
    root, run = tmp_path, tmp_path / "quality"
    run.mkdir()
    config_path = "neural_experiments/configs/h1-final-closure-v1.json"
    config = {"test": "synthetic receipt fixture"}
    source = {
        name: b"# synthetic committed source\n"
        for name in {
            *audit.AUDITOR_FILES,
            "neural_experiments/h1_final_closure.py",
            "neural_experiments/compression.py",
            "neural_experiments/H1_FINAL_CLOSURE_PROTOCOL.md",
        }
    }
    source[config_path] = (json.dumps(config) + "\n").encode()
    for name, contents in source.items():
        for base in (root, run / "source"):
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
    monkeypatch.setattr(
        audit.subprocess, "check_output", lambda args, cwd: source[args[-1].split(":", 1)[1]]
    )
    (run / "source.diff").write_text("")
    (run / "configuration.json").write_bytes(source[config_path])
    gates = {
        "historical_reference": True,
        "float32_equivalence": dict.fromkeys(audit.FLOAT_ARMS, True),
        "compression_quality": True,
    }
    write_json(run / "summary.json", {"gates": gates})
    write_json(run / "inventory.json", {})
    start = {
        "schema_version": 1,
        "kind": "h1_final_closure_quality",
        "status": "running",
        "run": str(run),
        "started_utc": "2026-10-24T20:00:00+00:00",
        "command": ["python", "quality"],
    }
    write_json(run / "start.json", start)
    manifest = {
        **start,
        "status": "complete",
        "finished_utc": "2026-10-24T20:00:01+00:00",
        "configuration": config,
        "config": {"path": config_path, "sha256": audit.digest(run / "configuration.json")},
        "summary_sha256": audit.digest(run / "summary.json"),
        "inventory_sha256": audit.digest(run / "inventory.json"),
        "source": {
            "head": "a" * 40,
            "files": {name: audit.digest(root / name) for name in source},
            "diff_sha256": audit.digest(run / "source.diff"),
        },
        "statuses": {
            "artifact_completion": "complete",
            "implementation_execution": "complete",
            "independent_verification": "not_run",
            "runtime_goals": "not_run",
        },
        "artifacts": {
            str(path.relative_to(run)): audit.digest(path)
            for path in run.rglob("*")
            if path.is_file()
        },
    }
    write_json(run / "manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        "kind": "h1_final_closure_quality_audit",
        "status": "passed",
        "run": str(run),
        "manifest_sha256": audit.digest(run / "manifest.json"),
        "summary_sha256": manifest["summary_sha256"],
        "config_sha256": manifest["config"]["sha256"],
        "source": manifest["source"],
        "artifact_sha256": manifest["artifacts"],
        "gates": gates,
        "qualified_arms": dict.fromkeys(audit.ARMS, True),
        "implementation_execution": "complete",
        "auditor_source": {name: audit.digest(root / name) for name in audit.AUDITOR_FILES},
    }
    path = root / "audit.json"
    write_json(path, receipt)
    return root, run, path, manifest, receipt


def test_receipt_requires_run_source_and_complete_artifacts(receipt_evidence):
    root, run, path, _, receipt = receipt_evidence
    assert audit.validate_audit_receipt(run, path, root) == receipt
    (run / "source" / audit.AUDITOR_FILES[0]).write_text("# changed retained source\n")
    with pytest.raises(ValueError, match="changed evidence"):
        audit.validate_audit_receipt(run, path, root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("run", "/another/quality"),
        ("manifest_sha256", "0" * 64),
        ("config_sha256", "0" * 64),
        ("implementation_execution", "complete_after_failure"),
    ],
)
def test_stale_mixed_or_failed_receipt_rejected(receipt_evidence, field, value):
    root, run, path, _, receipt = receipt_evidence
    receipt[field] = value
    write_json(path, receipt)
    with pytest.raises(ValueError):
        audit.validate_audit_receipt(run, path, root)


def test_failed_terminal_attempt_cannot_be_rehabilitated_by_receipt(receipt_evidence):
    root, run, path, manifest, receipt = receipt_evidence
    manifest["status"] = "failed"
    write_json(run / "manifest.json", manifest)
    receipt["manifest_sha256"] = audit.digest(run / "manifest.json")
    write_json(path, receipt)
    with pytest.raises(ValueError, match="not artifact-complete"):
        audit.validate_audit_receipt(run, path, root)


def test_changed_auditor_source_invalidates_runtime_admission(receipt_evidence):
    root, run, path, *_ = receipt_evidence
    (root / audit.AUDITOR_FILES[0]).write_text("# changed executable auditor\n")
    with pytest.raises(ValueError, match="changed evidence"):
        audit.validate_audit_receipt(run, path, root)


def test_runtime_and_quality_threshold_spellings_cannot_diverge():
    root = Path(__file__).resolve().parents[2]
    config = audit.read_json(root / "neural_experiments/configs/h1-final-closure-v1.json")
    audit.check_config(config)
    config["quality"]["parity_atol"] = 0.1
    with pytest.raises(ValueError, match="configuration changed"):
        audit.check_config(config)
