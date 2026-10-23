"""Independent arithmetic, causality, tamper and gate tests on synthetic data."""

import ast
import copy
import inspect
import json

import numpy as np
import pytest

from neural_experiments.v7 import audit, metrics, model, recoverability


def test_no_producer_imports():
    for node in ast.walk(ast.parse(inspect.getsource(audit))):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("neural_experiments.v7")
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith("neural_experiments.v7") for n in node.names)


def inputs(n=17000):
    rng = np.random.default_rng(7)
    c = 3
    features = {
        "x_tx3": rng.poisson(1, (n, c)).astype(np.float32),
        "x_tx45": rng.poisson(0.5, (n, c)).astype(np.float32),
        "x_sbp": rng.gamma(2, 1, (n, c)).astype(np.float32),
    }
    segments = [(0, min(n, 16000))] + ([(16000, n)] if n > 16000 else [])
    angle = np.arange(n) * 0.07
    labels = {
        "y": np.c_[np.cos(angle), np.sin(angle)].astype(np.float32),
        "eval_mask": rng.random(n) > 0.1,
    }
    return {
        "features": features,
        "segments": segments,
        **labels,
        "item": {
            "participant_id": "T2",
            "session_id": "T2-synthetic",
            "post_implant_day": 10,
            "n_channels": c,
            "n_bins": n,
            "segments": segments,
        },
    }


@pytest.mark.parametrize("method", audit.METHODS[1:])
def test_saved_fits_match_independent_normal_equations(tmp_path, method):
    case = inputs()
    raw = model.representation(case["features"], case["segments"], model.CONFIG)
    independent_raw = audit.reference_representation(
        case["features"], case["segments"], audit.CONFIG
    )
    np.testing.assert_allclose(raw, independent_raw, rtol=1e-12, atol=1e-12)
    fitted, receipt = model.fit(
        raw, case["y"], case["eval_mask"], case["segments"], 3, method, 16600
    )
    path = tmp_path / "model.npz"
    fitted.save(path)
    residual = audit.validate_fit(
        audit.numeric(path), receipt, independent_raw, case, case["segments"], 3, method, 16600
    )
    assert residual < 1e-10


@pytest.mark.parametrize("method", ["rolling120_local", "exponential60"])
def test_future_feature_and_label_changes_do_not_affect_fitting_reference(tmp_path, method):
    case = inputs()
    raw = model.representation(case["features"], case["segments"], model.CONFIG)
    fitted, receipt = model.fit(
        raw, case["y"], case["eval_mask"], case["segments"], 3, method, 16600
    )
    path = tmp_path / "model.npz"
    fitted.save(path)
    arrays = audit.numeric(path)
    raw[16500:] = np.nan
    case["y"][16500:] = np.nan
    audit.validate_fit(arrays, receipt, raw, case, case["segments"], 3, method, 16600)
    case["y"][1500:16500] *= -1
    with pytest.raises(ValueError, match="optimum"):
        audit.validate_fit(arrays, receipt, raw, case, case["segments"], 3, method, 16600)


@pytest.mark.parametrize("tamper", ["weights", "mean", "scale", "format_version"])
def test_rehashed_numeric_export_tamper_rejected(tmp_path, tamper):
    case = inputs()
    raw = model.representation(case["features"], case["segments"], model.CONFIG)
    fitted, receipt = model.fit(
        raw, case["y"], case["eval_mask"], case["segments"], 3, "rolling60", 16600
    )
    path = tmp_path / "model.npz"
    fitted.save(path)
    arrays = audit.numeric(path)
    if tamper == "format_version":
        arrays[tamper] = arrays[tamper].astype(float)
    else:
        arrays[tamper].flat[0] += 0.1
    with pytest.raises(ValueError, match="optimum|normalization|format"):
        audit.validate_fit(arrays, receipt, raw, case, case["segments"], 3, "rolling60", 16600)


def test_exponential_weighted_mean_loss_is_not_unweighted(tmp_path):
    case = inputs()
    raw = model.representation(case["features"], case["segments"], model.CONFIG)
    wrong, _ = model.fit(raw, case["y"], case["eval_mask"], case["segments"], 3, "expanding", 16600)
    _, receipt = model.fit(
        raw, case["y"], case["eval_mask"], case["segments"], 3, "exponential60", 16600
    )
    path = tmp_path / "wrong.npz"
    wrong.save(path)
    with pytest.raises(ValueError, match="optimum"):
        audit.validate_fit(
            audit.numeric(path), receipt, raw, case, case["segments"], 3, "exponential60", 16600
        )


def test_union_counts_only_requested_intervals_once():
    expected = {
        "frozen120": 120.0,
        "rolling60": 360.0,
        "rolling120": 420.0,
        "rolling120_local": 420.0,
        "rolling240": 540.0,
        "expanding": 540.0,
        "exponential60": 540.0,
    }
    for method, value in expected.items():
        intervals = [audit.fit_geometry(method, t)[:2] for t in audit.UPDATES]
        assert audit.unique_label_seconds(intervals) == value


def exported_case(tmp_path, n):
    case = inputs(n)
    if n > 16600:
        case["eval_mask"][16600:] = False
        case["eval_mask"][16600:16640] = True

    def export(method, start, stop, fitted, prediction, mask):
        model_path = tmp_path / f"{method}-{start}.model.npz"
        trace_path = tmp_path / f"{method}-{start}.trace.npz"
        if fitted is not None:
            fitted.save(model_path)
        np.savez(
            trace_path,
            prediction=prediction,
            eval_mask=mask,
            start_bin=np.asarray(start, np.int64),
            stop_bin=np.asarray(stop, np.int64),
        )
        return {
            "model_path": model_path.name if fitted is not None else None,
            "model_sha256": audit.sha256(model_path) if fitted is not None else None,
            "trace_path": trace_path.name,
            "trace_sha256": audit.sha256(trace_path),
        }

    rows, windows = recoverability.evaluate_session(case, save_artifact=export)
    raw = audit.reference_representation(case["features"], case["segments"], audit.CONFIG)
    return case, raw, rows, windows


def check(case, raw, rows, windows, root):
    return audit.check_session(
        case["item"],
        case,
        raw,
        audit.unique(rows, lambda r: (r["session_id"], r["method"])),
        audit.unique(windows, lambda r: (r["session_id"], r["method"], r["start_bin"])),
        root,
    )


@pytest.mark.parametrize("n", [1000, 13600, 13625, 17000])
def test_actual_producer_exports_audit_with_short_and_sparse_windows(tmp_path, n):
    case, raw, rows, windows = exported_case(tmp_path, n)
    rr, ww, errors, residuals = check(case, raw, rows, windows, tmp_path)
    assert len(rr) == 8 and len(ww) == 48
    assert max(errors, default=0) < 1e-10
    assert max(residuals, default=0) < 1e-10
    if n == 17000:
        assert rr[0]["metrics"]["n"] == ww[0]["metrics"]["n"]
        assert ww[1]["valid_rows"] == 40
        assert ww[1]["disposition"] == "insufficient_labels"


def test_rehashed_saved_trace_and_mask_tampering_rejected(tmp_path):
    case, raw, rows, windows = exported_case(tmp_path, 13700)
    window = next(w for w in windows if w["method"] == "rolling60" and w["start_bin"] == 13600)
    path = tmp_path / window["trace_path"]
    trace = audit.numeric(path)
    trace["prediction"][0] += 0.1
    np.savez(path, **trace)
    window["trace_sha256"] = audit.sha256(path)
    with pytest.raises(ValueError, match="replay"):
        check(case, raw, rows, windows, tmp_path)
    trace["eval_mask"][0] = not trace["eval_mask"][0]
    np.savez(path, **trace)
    window["trace_sha256"] = audit.sha256(path)
    with pytest.raises(ValueError, match="mask"):
        check(case, raw, rows, windows, tmp_path)


def aggregate_cases():
    rows, windows = [], []
    for person, count in (("T2", 8), ("T5", 2)):
        for day in range(count):
            for method in audit.METHODS:
                value = 0.2 if method in audit.ELIGIBLE else -0.1
                identity = {
                    "participant": person,
                    "session_id": f"{person}-{day}",
                    "post_implant_day": day,
                    "method": method,
                }
                metric = {"r2": value, "summed_mse": 1 - value, "angular_error_degrees": 70.0}
                rows.append(
                    {
                        **identity,
                        "metrics": metric,
                        "score_mask_sha256": "same",
                        "unique_requested_label_seconds": 100.0,
                    }
                )
                windows.extend(
                    {
                        **identity,
                        "start_bin": t,
                        "score_mask_sha256": "same",
                        "valid_rows": 100,
                        "metrics": metric.copy(),
                    }
                    for t in audit.UPDATES
                )
    return rows, windows


def test_latest_six_gate_bootstrap_equal_person_and_tie_order():
    rows, windows = aggregate_cases()
    expected = audit.summarize_reference(rows, windows)
    audit.compare_structure(metrics.summarize(rows, windows), expected)
    assert expected["primary_T2_session_ids"] == [f"T2-{i}" for i in range(2, 8)]
    assert expected["gate"]["passed"]
    assert expected["gate"]["selected_method"] == "rolling60"


def test_missing_primary_session_stays_in_six_denominator():
    rows, windows = aggregate_cases()
    for row in rows:
        if row["session_id"] == "T2-7":
            row["metrics"] = {"r2": None}
    expected = audit.summarize_reference(rows, windows)
    audit.compare_structure(metrics.summarize(rows, windows), expected)
    assert not expected["gate"]["passed"]
    assert expected["primary_T2"]["rolling60"]["selected_sessions"] == 6
    assert expected["primary_T2"]["rolling60"]["scored_sessions"] == 5


def test_mask_and_inventory_mismatches_fail_before_gate():
    rows, windows = aggregate_cases()
    bad = copy.deepcopy(rows)
    bad[-1]["score_mask_sha256"] = "different"
    with pytest.raises(ValueError, match="Unpaired"):
        audit.summarize_reference(bad, windows)
    with pytest.raises(ValueError, match="inventory"):
        audit.summarize_reference(rows, windows[:-1])


def test_undefined_r2_window_still_counts_mse_harm_time():
    rows, windows = aggregate_cases()
    for w in windows:
        if w["participant"] == "T2" and w["method"] == "rolling60":
            w["metrics"] = {"r2": None, "summed_mse": 2.0, "angular_error_degrees": 90.0}
    expected = audit.summarize_reference(rows, windows)
    audit.compare_structure(metrics.summarize(rows, windows), expected)
    assert (
        expected["methods"]["rolling60"]["participants"][0][
            "fraction_valid_time_in_windows_worse_than_zero"
        ]
        == 1.0
    )


def metadata_fixture(tmp_path, monkeypatch):
    people = ["T2", "T3", "T5", "T6", "T7", "T8", "T9", "T10", "T11"]
    parent_refs, records, pins = {}, [], {}
    for role in ("development", "new_sessions"):
        path = tmp_path / role / "index.json"
        path.parent.mkdir()
        items = []
        for person in people:
            ranks = (
                range(8)
                if person == "T3" and role == "development"
                else (
                    range(15)
                    if role == "development"
                    else (range(15, 18) if person != "T3" else [])
                )
            )
            for rank in ranks:
                sid = f"{person}-{rank}"
                item = {
                    "session_id": sid,
                    "participant_id": person,
                    "post_implant_day": rank,
                    "role": role,
                    "n_bins": 100,
                    "bin_seconds": 0.02,
                    "segments": [(0, 100)],
                    "feature_path": sid + ".features.npz",
                    "target_path": sid + ".targets.npz",
                    "feature_sha256": "f" * 64,
                    "target_sha256": "e" * 64,
                    "reused_parent_artifacts": {
                        "feature": {"path": "grandparent-feature"},
                        "target": {"path": "grandparent-target"},
                    },
                }
                items.append(item)
        path.write_text(
            json.dumps({"role": role, "feature_definition": {"synthetic": True}, "sessions": items})
        )
        pin = audit.sha256(path)
        parent_refs[role], pins[role] = {"path": str(path), "sha256": pin}, pin
        for item in items:
            record = copy.deepcopy(item)
            record["role"] = "development"
            record["v7_parent"] = {
                "index_path": str(path),
                "index_sha256": pin,
                "role": role,
                "session_id": item["session_id"],
            }
            record["reused_parent_artifacts"] = {
                kind: {
                    "path": str(path.parent / item[kind + "_path"]),
                    "sha256": item[kind + "_sha256"],
                    "mode": "immutable_hard_link",
                }
                for kind in ("feature", "target")
            }
            records.append(record)
    index = {
        "schema_version": 1,
        "role": "development",
        "participants": people,
        "provider": {"doi": "10.5061/dryad.x0k6djj1h", "version_id": 462702},
        "feature_definition": {"synthetic": True},
        "parents": parent_refs,
        "sessions": records,
        "exposure": {
            "rule": "exact_union_of_exposed_v6_development_and_new_session_prefixes",
            "sessions": 152,
            "maximum_bins_per_session": 30000,
            "continuation_tails_included": False,
            "fresh_data_included": False,
        },
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))
    monkeypatch.setattr(audit, "PARENT_PINS", pins)
    return path, index


def test_immediate_parent_artifact_binding_replaces_grandparent_metadata(tmp_path, monkeypatch):
    path, _ = metadata_fixture(tmp_path, monkeypatch)
    assert len(audit.Inputs(path).items) == 152


@pytest.mark.parametrize("field", ["reused_parent_artifacts", "n_bins"])
def test_rehashed_current_index_cannot_change_bound_parent_values(tmp_path, monkeypatch, field):
    path, index = metadata_fixture(tmp_path, monkeypatch)
    if field == "n_bins":
        index["sessions"][0][field] = 101
    else:
        index["sessions"][0][field]["feature"]["path"] = "grandparent-feature"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="parent"):
        audit.Inputs(path)
