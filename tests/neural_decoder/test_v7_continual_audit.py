"""Synthetic producer-to-independent-auditor tests for the amended V7B study."""

import ast
import copy
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from neural_experiments.v7 import audit_continual as audit
from neural_experiments.v7 import continual, continual_metrics, model


def test_only_independent_reference_imports():
    for node in ast.walk(ast.parse(inspect.getsource(audit))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "neural_experiments"
        ):
            assert "audit" in node.module.split(".")[-1]


def case(n):
    rng = np.random.default_rng(71)
    channels = 2
    features = {
        "x_tx3": rng.poisson(2, (n, channels)).astype(np.float32),
        "x_tx45": rng.poisson(1, (n, channels)).astype(np.float32),
        "x_sbp": rng.gamma(2, 1, (n, channels)).astype(np.float32),
    }
    segments = [(0, n)] if n <= 17000 else [(0, 17000), (17000, n)]
    angle = np.arange(n) * 0.03
    return {
        "features": features,
        "segments": segments,
        "y": np.c_[np.cos(angle), np.sin(angle)].astype(np.float32),
        "eval_mask": rng.random(n) > 0.1,
        "item": {
            "participant_id": "T2",
            "session_id": "T2-synthetic",
            "post_implant_day": 1,
            "n_bins": n,
            "n_channels": channels,
            "segments": segments,
        },
    }


def exported(tmp_path, n, threshold=0.25):
    source = case(n)

    def save(kind, method, ordinal, payload):
        path = tmp_path / f"{method}-{ordinal}-{kind}.npz"
        if kind == "model":
            payload["decoder"].save(path)
            return {
                "decoder": model.Decoder.load(path),
                "record": {"model_path": path.name, "model_sha256": audit.sha256(path)},
            }
        np.savez(path, **payload)
        return {"trace_path": path.name, "trace_sha256": audit.sha256(path)}

    rows, events = continual.evaluate_session(source, threshold, save_artifact=save)
    # The runner binds the threshold run identity onto every control and event.
    rows = [{**r, "threshold": threshold} for r in rows]
    events = [{**e, "threshold": threshold} for e in events]
    raw = audit.reference_representation(source["features"], source["segments"], audit.CONFIG)
    return source, raw, rows, events


@pytest.mark.parametrize("n", [1000, 1500, 1501, 7600, 8000, 10600, 12150, 30000])
def test_actual_producer_ledgers_and_traces_reconstruct(tmp_path, n):
    source, raw, rows, events = exported(tmp_path, n)
    for row in rows:
        selected = [e for e in events if e["method"] == row["method"]]
        _, error, residuals = audit.check_case(
            source["item"], source, raw, row["method"], 0.25, row, selected, tmp_path
        )
        assert error < 1e-10
        assert max(residuals, default=0) < 1e-10
    if n <= 1500:
        assert events == []
        assert all(
            r["requested_probes"] == 0 and r["unique_requested_label_seconds"] == 0 for r in rows
        )
    if n == 30000:
        for row in rows:
            if row["method"] in ("periodic240", "triggered240"):
                assert row["available_label_seconds"] == 240.0
                assert row["interruption_seconds"] == 128.0


def test_drift_evidence_is_prefix_only_and_matches_independent_KL():
    source = case(30000)
    raw = model.representation(source["features"], source["segments"], model.CONFIG)
    expected = audit.drift_reference(raw, 2, 16500)
    audit.compare_structure(continual.drift_evidence(raw, 2, 16500), expected)
    raw[16500:] = np.nan
    audit.compare_structure(continual.drift_evidence(raw, 2, 16500), expected)


@pytest.mark.parametrize("threshold", audit.THRESHOLDS)
def test_neural_only_schedule_forces_fixed_four_probe_budget(threshold):
    used, last_end = 0, 7500
    chosen = []
    for i, start in enumerate(audit.SLOTS):
        actual = continual.choose_probe("triggered240", i, used, last_end, 0.0, threshold)
        expected = audit.decision_reference("triggered240", i, used, last_end, 0.0, threshold)
        audit.compare_structure(actual, expected)
        if expected["request_probe"]:
            chosen.append(start // 50)
            used += 1
            last_end = start + 1500
    assert chosen == [330, 390, 450, 510]


@pytest.mark.parametrize("kind", ["prediction", "interruption_mask", "purchased_mask"])
def test_rehashed_trace_tampering_rejected(tmp_path, kind):
    source, raw, rows, events = exported(tmp_path, 12500)
    row = next(r for r in rows if r["method"] == "periodic240")
    path = tmp_path / row["trace_path"]
    trace = audit.numeric(path)
    if kind == "prediction":
        trace[kind][11000] += 0.1
    else:
        trace[kind][11000] = not trace[kind][11000]
    np.savez(path, **trace)
    row["trace_sha256"] = audit.sha256(path)
    with pytest.raises(ValueError, match="mask|prediction"):
        audit.check_case(
            source["item"],
            source,
            raw,
            "periodic240",
            0.25,
            row,
            [e for e in events if e["method"] == "periodic240"],
            tmp_path,
        )


def test_controller_evidence_and_model_binding_tamper_rejected(tmp_path):
    source, raw, rows, events = exported(tmp_path, 30000)
    row = next(r for r in rows if r["method"] == "triggered240")
    selected = [e for e in events if e["method"] == "triggered240"]
    bad = copy.deepcopy(selected)
    next(e for e in bad if e["event"] == "probe_decision")["evidence"]["kl_mean"] += 1
    with pytest.raises(ValueError, match="aggregate"):
        audit.check_case(source["item"], source, raw, "triggered240", 0.25, row, bad, tmp_path)
    fitted = next(e for e in selected if e["event"] == "model_fitted")
    path = tmp_path / fitted["model_path"]
    arrays = audit.numeric(path)
    arrays["weights"][0, 0] += 0.1
    np.savez(path, **arrays)
    fitted["model_sha256"] = audit.sha256(path)
    with pytest.raises(ValueError, match="optimum"):
        audit.check_case(source["item"], source, raw, "triggered240", 0.25, row, selected, tmp_path)


def aggregate_rows():
    rows = []
    for person in ("T2", "T5", "T6", "T7", "T8", "T9", "T10", "T11"):
        for day in range(3):
            for method, value in zip(audit.METHODS, (-0.1, 0.1, 0.15, 0.2, 0.23), strict=True):
                metric = {"r2": value, "summed_mse": 1 - value, "angular_error_degrees": 60.0}
                rows.append(
                    {
                        "participant": person,
                        "session_id": f"{person}-{day}",
                        "method": method,
                        "threshold": 0.25,
                        "n_bins": 30000,
                        "score_mask_sha256": "same",
                        "metrics": metric,
                        "planned_label_seconds": 240,
                        "available_label_seconds": 240.0,
                        "interruption_seconds": 128.0,
                        "diagnostic_windows": [{"valid_rows": 100, "metrics": metric.copy()}],
                    }
                )
    return rows


def test_whole_person_source_threshold_selection():
    rows = [
        {**copy.deepcopy(r), "threshold": t} for r in aggregate_rows() for t in audit.THRESHOLDS
    ]
    expected = audit.selection_reference(rows)
    audit.compare_structure(continual_metrics.select_source(rows), expected)
    assert all(c["threshold"] == 0.1 for c in expected["choices"])
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["participant"] == "T2" and row["threshold"] == 1.0:
            row["metrics"]["r2"] = 100.0
    modified = audit.selection_reference(changed)
    assert modified["choices"][0] == expected["choices"][0]
    assert modified["choices"][1]["threshold"] == 1.0


def test_independent_full_interval_pairing_bootstrap_and_gates():
    rows = aggregate_rows()
    expected = audit.summary_reference(rows)
    audit.compare_structure(continual_metrics.summarize(rows), expected)
    assert all(expected["gates"].values())
    rows[-1]["available_label_seconds"] = 210.0
    expected = audit.summary_reference(rows)
    audit.compare_structure(continual_metrics.summarize(rows), expected)
    assert not expected["gates"]["triggered_timing_value"]


def test_short_primary_case_cannot_disappear_from_denominator():
    rows = aggregate_rows()
    for row in rows:
        if row["session_id"] == "T2-2":
            row["metrics"] = {"r2": None}
    expected = audit.summary_reference(rows)
    audit.compare_structure(continual_metrics.summarize(rows), expected)
    assert not expected["gates"]["periodic_T2_recovery"]
    assert expected["T2"]["periodic240"]["selected_sessions"] == 3
    assert expected["T2"]["periodic240"]["scored_sessions"] == 2


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fresh_metadata(tmp_path, monkeypatch, selected_count=3):
    """A tiny metadata-only archive walk; no real sample arrays are accessed."""
    plan = write_json(tmp_path / "plan.json", {"locked": "study"})
    acquire = write_json(tmp_path / "acquire.json", {"locked": "acquisition"})
    parent = write_json(tmp_path / "parent.json", {"locked": "parent"})
    monkeypatch.setattr(audit, "PLAN_SHA", audit.sha256(plan))
    monkeypatch.setattr(audit, "ACQUISITION_SHA", audit.sha256(acquire))
    monkeypatch.setattr(
        audit, "PARENT_PINS", dict.fromkeys(("development", "new_sessions"), audit.sha256(parent))
    )
    people = ["T2", "T3"]
    archive = {"file_id": 123, "provider_sha256": "a" * 64}
    old, selected, seen = [], [], []
    for rank in range(1, 19 + selected_count):
        sid = f"T2_day_{rank}_decoding"
        raw = tmp_path / "raw" / "T2" / f"{sid}.mat"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(f"synthetic raw {rank}".encode())
        member = {
            "participant_id": "T2",
            "session_id": sid,
            "post_implant_day": rank,
            "archive_ordinal": rank - 1,
            "regular_session_rank": rank,
            "member_name": f"decoding/T2/{sid}.mat",
            "member_size": raw.stat().st_size,
            "member_sha256": audit.sha256(raw),
            "raw_path": f"raw/T2/{sid}.mat",
        }
        (old if rank <= 18 else selected).append(member)
        seen.append(
            {
                "ordinal": rank - 1,
                "name": member["member_name"],
                "size": member["member_size"],
                "regular_file": True,
            }
        )
    receipt = {
        "status": "complete",
        "participant_id": "T2",
        "plan_sha256": audit.ACQUISITION_SHA,
        "selection": {
            "regular_session_ranks_one_based": [19, 20, 21],
            "scores_or_duration_used_for_selection": False,
        },
        "v6_development_index_sha256": audit.sha256(parent),
        "v6_new_index_sha256": audit.sha256(parent),
        "archive": archive,
        "full_archive_sha256_verified": False,
        "verified_prior_members": old,
        "selected": selected,
        "seen_members": seen,
        "tar_end_observed": selected_count < 3,
        "missing_requested_ranks": list(range(19 + selected_count, 22)),
    }
    path = write_json(tmp_path / "acquisition" / "T2.json", receipt)

    def extend(member):
        return {
            **member,
            "archive_file_id": 123,
            "archive_sha256_published": "a" * 64,
            "full_archive_sha256_verified": False,
        }

    development = SimpleNamespace(
        index={
            "participants": people,
            "provider": {"pinned": True},
            "feature_definition": {"fixed": True},
        },
        items={m["session_id"]: {"participant_id": "T2", "source": extend(m)} for m in old},
    )
    index = {
        "schema_version": 1,
        "role": "fresh",
        "participants": ["T2"],
        "provider": {"pinned": True},
        "feature_definition": {"fixed": True},
        "acquisition_plan_sha256": audit.ACQUISITION_SHA,
        "acquisition_lock": {"path": str(acquire), "sha256": audit.ACQUISITION_SHA},
        "study_lock": {"path": str(plan), "sha256": audit.PLAN_SHA},
        "v6_development_index_path": str(parent),
        "v6_development_index_sha256": audit.sha256(parent),
        "v6_new_index_path": str(parent),
        "v6_new_index_sha256": audit.sha256(parent),
        "acquisition_sources": [{"path": "acquisition/T2.json", "sha256": audit.sha256(path)}],
        "participant_dispositions": [
            {
                "participant_id": "T2",
                "selected_sessions": selected_count,
                "missing_requested_ranks": receipt["missing_requested_ranks"],
            }
        ],
        "sessions": [
            {
                "session_id": m["session_id"],
                "source": extend(m),
                "role": "fresh",
                "participant_id": "T2",
                "post_implant_day": m["post_implant_day"],
                "n_bins": 100,
                "full_prepared_n_bins": 100,
                "bin_seconds": 0.02,
                "raw_source_root": str(tmp_path),
                "segments": [[0, 100]],
            }
            for m in selected
        ],
    }
    return write_json(tmp_path / "index.json", index), development, index, receipt


@pytest.mark.parametrize("selected_count", [0, 2, 3])
def test_fresh_rank_inventory_retains_archive_end_shortages(tmp_path, monkeypatch, selected_count):
    path, development, _, _ = fresh_metadata(tmp_path, monkeypatch, selected_count)
    inputs = audit.FreshInputs(path, development)
    assert len(inputs.items) == selected_count


@pytest.mark.parametrize("tamper", ["prior_hash", "rank", "extra_source_field", "shortage", "raw"])
def test_rehashed_fresh_provenance_tamper_rejected(tmp_path, monkeypatch, tamper):
    path, development, index, receipt = fresh_metadata(tmp_path, monkeypatch, 2)
    if tamper == "prior_hash":
        receipt["verified_prior_members"][0]["member_sha256"] = "b" * 64
    elif tamper == "rank":
        receipt["selected"][0]["regular_session_rank"] = 20
    elif tamper == "extra_source_field":
        index["sessions"][0]["source"]["unbound"] = True
    elif tamper == "shortage":
        receipt["tar_end_observed"] = False
    else:
        (tmp_path / receipt["selected"][0]["raw_path"]).write_bytes(b"changed")
    receipt_path = write_json(tmp_path / "acquisition" / "T2.json", receipt)
    index["acquisition_sources"][0]["sha256"] = audit.sha256(receipt_path)
    write_json(path, index)
    with pytest.raises(ValueError):
        audit.FreshInputs(path, development)


def source_publication(tmp_path):
    rows = [
        {**copy.deepcopy(r), "threshold": t} for r in aggregate_rows() for t in audit.THRESHOLDS
    ]
    selection = audit.selection_reference(rows)
    original = {
        "status": "complete",
        "phase": "source",
        "index": {"sha256": "index"},
        "code_sha256": {"code.py": "source"},
        "rows": rows,
        "selection": selection,
    }
    source = write_json(tmp_path / "source" / "manifest.json", original)
    passed = {
        "status": "PASS",
        "stage": "source",
        "manifest_sha256": audit.sha256(source),
        "plan_sha256": audit.PLAN_SHA,
        "index_sha256": "index",
        "code_sha256": original["code_sha256"],
        "selection": selection,
    }
    receipt = write_json(tmp_path / "source-audit.json", passed)
    published = {
        "schema_version": 1,
        "plan_sha256": audit.PLAN_SHA,
        "source_manifest_sha256": audit.sha256(source),
        "source_audit_sha256": audit.sha256(receipt),
        "source_audit_status": "PASS",
        "scientific_code_sha256": original["code_sha256"],
        "selection": selection,
    }
    publication = write_json(tmp_path / "public.json", published)
    manifest = {
        "code_sha256": original["code_sha256"],
        "source_selection": {"path": "public.json", "sha256": audit.sha256(publication)},
    }
    return manifest, source, receipt, publication


def test_fresh_source_choice_binds_actual_passed_run_and_published_digest(tmp_path):
    manifest, source, receipt, _ = source_publication(tmp_path)
    selected, _ = audit.source_binding(manifest, tmp_path, source.parent, receipt)
    assert len(selected["choices"]) == 8
    with pytest.raises(ValueError, match="explicit retained"):
        audit.source_binding(manifest, tmp_path, None, receipt)


@pytest.mark.parametrize("tamper", ["source", "audit", "publication", "code"])
def test_published_source_claim_alone_does_not_authorize_assessment(tmp_path, tamper):
    manifest, source, receipt, publication = source_publication(tmp_path)
    if tamper == "source":
        value = audit.read(source)
        value["rows"][0]["metrics"]["r2"] += 1
        write_json(source, value)
    elif tamper == "audit":
        value = audit.read(receipt)
        value["status"] = "FAIL"
        write_json(receipt, value)
    elif tamper == "publication":
        value = audit.read(publication)
        value["selection"]["choices"][0]["threshold"] = 1.0
        write_json(publication, value)
        manifest["source_selection"]["sha256"] = audit.sha256(publication)
    else:
        manifest["code_sha256"] = {"code.py": "changed"}
    with pytest.raises(ValueError):
        audit.source_binding(manifest, tmp_path, source, receipt)
