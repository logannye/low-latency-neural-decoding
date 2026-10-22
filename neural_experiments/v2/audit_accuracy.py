"""Independent retained-trace audit of the H1 V2 run; no training or timing."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retained_plan(run, expected_hash):
    matches = sorted(
        path for path in (Path(run) / "source").rglob("*.json") if digest(path) == expected_hash
    )
    if not matches:
        raise ValueError("no retained plan matches the manifest hash")
    return read(matches[0])


def audit(run, index_path):
    run, index_path = Path(run), Path(index_path)
    manifest = read(run / "manifest.json")
    assert manifest["status"] == "complete"
    plan = retained_plan(run, manifest["plan_sha256"])
    summary = read(run / "summary.json")
    assert digest(run / "summary.json") == manifest["summary_sha256"]
    artifacts = read(run / "artifact-hashes.json")
    assert digest(run / "artifact-hashes.json") == manifest["artifacts_sha256"]
    for relative, expected in artifacts.items():
        assert digest(run / relative) == expected, relative
    assert digest(index_path) == manifest["index_sha256"]
    index = read(index_path)
    targets = {}
    for item in index["files"]:
        assert item["day"] in {f"S{i}" for i in range(5)}
        target_path = index_path.parent / item["targets"]
        assert digest(target_path) == item["targets_sha256"]
        with np.load(target_path, allow_pickle=False) as data:
            targets[item["file_id"]] = (data["targets"].astype(np.float64), data["eval_mask"])
    errors, labels, score_rows = [], [], 0
    rows = [(row, 0) for row in summary["main"] + summary["diagnostics"]]
    rows += [(row, 1500) for row in summary["calibration"]]
    for arm in summary["schedule_decision"]["results"]:
        rows += [(checkpoint, 0) for checkpoint in arm["checkpoints"].values()]
    for row, suffix_start in rows:
        truth, predicted = [], []
        for file_id, evidence in row["predictions"].items():
            path = run / evidence["path"]
            assert digest(path) == evidence["sha256"]
            prediction = np.load(path, allow_pickle=False).astype(np.float64)
            y, original_mask = targets[file_id]
            assert prediction.shape == y.shape and np.isfinite(prediction).all()
            mask = original_mask & (np.arange(len(y)) >= suffix_start)
            if suffix_start:
                boundary = round(row["budget_seconds"] / 0.02)
                assert evidence["prefix_bins"] == boundary
                assert evidence["suffix_scored_bins"] == int(mask.sum())
                expected_labels = (
                    int(original_mask[:boundary].sum()) if row["method"] != "unadapted" else 0
                )
                assert evidence["prefix_scored_labels"] == expected_labels
                labels.append(expected_labels)
                base = np.load(path.parents[2] / f"{file_id}.npy", allow_pickle=False).astype(
                    np.float64
                )
                np.testing.assert_array_equal(prediction[:boundary], base[:boundary])
                if row["method"] == "unadapted" or boundary == 0:
                    np.testing.assert_array_equal(prediction, base)
            truth.append(y[mask])
            predicted.append(prediction[mask])
        y, prediction = np.concatenate(truth), np.concatenate(predicted)
        expected = r2_score(y, prediction, multioutput="variance_weighted")
        errors.append(abs(expected - row["scores"]["r2_mean"]))
        assert errors[-1] < 1e-12
        per_output = r2_score(y, prediction, multioutput="raw_values")
        saved_day = next(iter(row["scores"]["per_day"].values()))
        np.testing.assert_allclose(per_output, saved_day["r2_per_output"], atol=1e-12, rtol=0)
        assert len(y) == row["scores"]["num_scored_bins"]
        score_rows += 1
    for row in summary["diagnostics"]:
        exposure = row["exposure"]
        assert row["training_loss_rows"] == exposure["eligible_labels_each"]
        sources = exposure[f"{row['condition']}_sources"]
        assert sum(s["stop"] - s["start"] for s in sources) == exposure["raw_bins_each"]
        assert not set(row["predictions"]) & {s["file_id"] for s in sources}
        expected_labels = 0
        for source in sources:
            mask = targets[source["file_id"]][1][source["start"] : source["stop"]]
            expected_labels += int(mask[62:].sum())
        assert expected_labels >= exposure["eligible_labels_each"]
    selected = summary["selection"]["runtime_candidate"]
    assert selected["seed"] == 0 and selected["day"] == "S4"
    baseline = {
        (r["seed"], r["day"]): r["scores"]["r2_mean"]
        for r in summary["main"]
        if r["arm"] == "rf15_standardized"
    }
    eligible = []
    for order, arm in enumerate(plan["arms"]):
        records = [r for r in summary["main"] if r["arm"] == arm["name"]]
        assert len(records) == 9
        gains = [r["scores"]["r2_mean"] - baseline[r["seed"], r["day"]] for r in records]
        day_gains = [
            np.mean([g for r, g in zip(records, gains, strict=True) if r["day"] == day])
            for day in ("S2", "S3", "S4")
        ]
        if np.mean(gains) >= 0.005 and min(day_gains) >= -0.01:
            eligible.append(
                (
                    -np.mean([r["scores"]["r2_mean"] for r in records]),
                    records[0]["parameter_count"],
                    order,
                    arm["name"],
                )
            )
    independently_selected = min(eligible)[-1] if eligible else "rf15_standardized"
    assert selected["arm"] == independently_selected
    return {
        "status": "pass",
        "audit_source_sha256": digest(Path(__file__)),
        "run_summary_sha256": manifest["summary_sha256"],
        "artifact_inventory_sha256": manifest["artifacts_sha256"],
        "checked_artifacts": len(artifacts),
        "recomputed_score_rows": score_rows,
        "maximum_r2_absolute_error": max(errors),
        "selected_arm": independently_selected,
        "checks": [
            "all artifact hashes",
            "retained plan located by manifest hash",
            "development-only input and target hashes",
            "independent sklearn VW and per-output R2",
            "common calibration suffix and exact prefix label counts",
            "unadapted prefix and zero-budget identity",
            "diagnostic exposure counts and training/evaluation file exclusion",
            "independent candidate gate",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("preserve existing audit receipts; choose a fresh output path")
    with threadpool_limits(limits=1):
        receipt = audit(arguments.run, arguments.index)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
