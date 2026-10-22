"""Synthetic parity and adversarial checks for the separate V6 verifier."""

import ast
import copy
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from neural_experiments.v6 import audit, model, policy


def test_no_producer_imports_in_audit():
    tree = ast.parse(inspect.getsource(audit))
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in imports:
        if isinstance(node, ast.ImportFrom) and node.module in (
            "neural_experiments.v5",
            "neural_experiments.v6",
        ):
            assert all(name.name == "audit" for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("neural_experiments.v6")


def history(values=()):
    return {
        "current_session_id": "new",
        "current_post_implant_day": 10,
        "past_learning_curves": [
            {
                "session_id": f"old{i}",
                "post_implant_day": i,
                "budgets": {
                    str(b): {
                        "r2": r,
                        "n": 1000,
                        "coordinate_mse": (1 - r) / 2 if r is not None else None,
                    }
                    for b, r in zip((120, 240, 360), rr, strict=True)
                },
            }
            for i, rr in enumerate(values)
        ],
    }


def stage(budget, gain=0.1):
    end, start = 4500 + budget * 50, 1600 + budget * 50
    angle = np.arange(end) * 2 * np.pi / 100
    y = np.column_stack((np.cos(angle), np.sin(angle)))
    mask = np.ones(end, bool)
    mask[:1500] = False
    segments = [{"start": 0, "end": start + 103}, {"start": start + 103, "end": end}]
    prediction = y * (1 - np.sqrt(1 - 2 * gain))
    expected = audit.confirmation_reference(y, mask, segments, start, end, prediction, np.zeros(2))
    actual = policy.confirmation_evidence(y, mask, segments, start, end, prediction, np.zeros(2))
    audit.compare_structure(actual, expected)
    return {
        "training_seconds": budget,
        "requested_label_seconds": budget + 60,
        "available_label_seconds": budget + 60,
        "decision_time_seconds": budget + 90,
        "candidate_id": "tx45_sbp_prior0" if budget == 0 else audit.FIXED,
        "model_sha256": f"{budget:064x}",
        "fit_stop_bin": 1500 + budget * 50,
        "confirmation_start_bin": start,
        "confirmation_stop_bin": end,
        "confirmation": expected,
    }


@pytest.mark.parametrize("budget", audit.BUDGETS)
def test_confirmation_direct_arithmetic_matches_exported_policy(budget):
    result = stage(budget)
    start = result["confirmation_start_bin"]
    assert all(
        not (r["start_bin"] < start + 103 < r["stop_bin"]) for r in result["confirmation"]["groups"]
    )


@pytest.mark.parametrize("config", audit.grids()[1])
def test_independent_sequential_reference_matches_lazy_decision(config):
    stages = {b: stage(b, 0.1 if b >= 120 else 0) for b in audit.BUDGETS}
    calls = []

    def callback(b):
        calls.append(b)
        return stages[b]

    actual = policy.decide(history(), callback, config)
    expected, requested = audit.decision_reference(history(), stages, config)
    audit.compare_structure(actual, expected)
    assert calls == requested == [0, 60, 120]


@pytest.mark.parametrize(
    "values",
    [[], [(-0.1, 0.1, 0.12)] * 2, [(-0.1, -0.2, 0.1)] * 3, [(None, None, None), (-0.1, 0.1, 0.2)]],
)
def test_independent_floor_matches_without_query_outcomes(values):
    h = history(values)
    audit.compare_structure(policy.budget_floor(h), audit.floor_reference(h))


def test_current_session_cannot_enter_floor():
    h = history([(-0.1, 0.1, 0.2)] * 2)
    h["past_learning_curves"][0]["session_id"] = "new"
    with pytest.raises(ValueError, match="Future/current"):
        audit.floor_reference(h)


def fixture(tmp_path):
    rng = np.random.default_rng(6)
    n, channels, stop = 5000, 3, 4500
    raw = rng.normal(size=(n, channels * 2))
    angle = np.arange(n) * 0.04
    labels = {
        "y": np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32),
        "eval_mask": np.ones(n, bool),
    }
    segments = [{"start": 0, "end": 3000}, {"start": 3000, "end": n}]
    prior = rng.normal(0, 0.01, size=(channels * 2 + 1, 2))
    candidate = next(c for c in model.candidate_grid() if c["candidate_id"] == audit.FIXED)
    fitted, receipt = model.fit(
        raw,
        labels["y"],
        labels["eval_mask"],
        [(s["start"], s["end"]) for s in segments],
        channels,
        stop,
        candidate,
        prior,
    )
    path = tmp_path / "model.npz"
    fitted.save(path)
    item = {
        "participant_id": "p",
        "session_id": "p9",
        "post_implant_day": 9,
        "n_channels": channels,
        "n_bins": n,
        "segments": segments,
    }
    row = {
        "participant": "p",
        "session_id": "p9",
        "phase": "frozen_full",
        "training_seconds": 60,
        "candidate": candidate,
        "fit": receipt,
        "history": [],
        "model_path": path.name,
        "model_sha256": audit.sha256(path),
    }
    verifier = object.__new__(audit.Verifier)
    verifier.root, verifier.residuals, verifier.models_checked, verifier.fits_checked = (
        tmp_path,
        [],
        0,
        0,
    )
    return SimpleNamespace(
        verifier=verifier,
        item=item,
        row=row,
        raw={"tx45_sbp": raw},
        prior={"tx45_sbp": prior},
        history={"tx45_sbp": []},
        labels=labels,
        path=path,
    )


def test_saved_fixed_fit_satisfies_independent_normal_equations(tmp_path):
    f = fixture(tmp_path)
    f.verifier.fit_v6(f.row, f.item, f.raw, f.prior, f.history, f.labels)
    assert max(f.verifier.residuals) < 1e-10


def test_future_label_perturbation_leaves_fit_valid_but_training_changes_do_not(tmp_path):
    f = fixture(tmp_path)
    f.labels["y"][4500:] *= -1
    f.verifier.fit_v6(f.row, f.item, f.raw, f.prior, f.history, f.labels)
    f.labels["y"][1500:4500] *= -1
    with pytest.raises(ValueError, match="optimum"):
        f.verifier.fit_v6(f.row, f.item, f.raw, f.prior, f.history, f.labels)


@pytest.mark.parametrize("field", ["weights", "mean"])
def test_rehashed_numeric_tamper_rejected(tmp_path, field):
    f = fixture(tmp_path)
    arrays = audit.numeric(f.path)
    arrays[field].flat[0] += 0.1
    np.savez(f.path, **arrays)
    f.row["model_sha256"] = audit.sha256(f.path)
    with pytest.raises(ValueError, match="optimum|normalization"):
        f.verifier.fit_v6(f.row, f.item, f.raw, f.prior, f.history, f.labels)


def test_normalizer_receipt_cannot_claim_later_available_data(tmp_path):
    f = fixture(tmp_path)
    f.row["fit"]["normalization_stop_bin"] = 5000
    with pytest.raises((ValueError, AssertionError), match="normalizer"):
        f.verifier.fit_v6(f.row, f.item, f.raw, f.prior, f.history, f.labels)


def test_source_selection_independent_equal_person_reference():
    rows = [
        {
            "participant": person,
            "session_id": f"{person}-{i}",
            "policy_id": f"c{j}",
            "action": "accept",
            "label_seconds": 420 if i == 2 else 60 + j * 60,
            "daily240_r2": None if i == 2 else 0.3,
            "zero_mse": None if i == 2 else 0.5,
            "metrics": {
                "r2": None if i == 2 else 0.3 - j * 0.005,
                "summed_mse": None if i == 2 else 0.7 + j * 0.005,
            },
        }
        for person in ("a", "b", "c")
        for i in range(3)
        for j in range(4)
    ]
    expected = audit.selection_reference(rows, ["a", "b", "c"])
    audit.compare_structure(
        policy.choose_policy(rows, policy.policy_grid(), ["a", "b", "c"]), expected
    )
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["participant"] == "a":
            row["label_seconds"] = 420
            if row["metrics"]["r2"] is not None:
                row["metrics"] = {"r2": -1.0, "summed_mse": 2.0}
    assert audit.selection_reference(changed, ["a", "b", "c"])[0] == expected[0]


def test_production_source_case_all_artifacts_pass_separate_audit(tmp_path, monkeypatch):
    """Synthetic full producer fit/export/confirmation/V5 path, no real outcomes."""
    from neural_experiments.v6 import study

    n, channels = 23000, 2
    rng = np.random.default_rng(662)
    features = {
        k: rng.poisson(1, (n, channels)).astype(np.float32) for k in ("x_tx3", "x_tx45", "x_sbp")
    }
    features["channel_ids"] = np.arange(channels, dtype=np.int64)
    angle = np.arange(n) * 0.023
    labels = {
        "y": np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32),
        "eval_mask": np.ones(n, bool),
    }
    target = tmp_path / "targets.npz"
    np.savez(target, **labels)
    item = {
        "participant_id": "p",
        "session_id": "p9",
        "post_implant_day": 9,
        "n_channels": channels,
        "n_bins": n,
        "segments": [{"start": 0, "end": 12000}, {"start": 12000, "end": n}],
        "target_path": target.name,
        "target_sha256": audit.sha256(target),
    }
    case = {
        "item": item,
        "features": features,
        "segments": [(s["start"], s["end"]) for s in item["segments"]],
        **labels,
    }
    for module in (study, study.old):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "relative", lambda p: str(p.resolve().relative_to(tmp_path)))
        monkeypatch.setattr(module, "load_case", lambda *_a, **_k: case)
    output = study.init_output(tmp_path / "run")
    index_path = tmp_path / "index.json"
    layouts, stats = {"p9": tuple(features["channel_ids"])}, {f: {} for f in audit.FAMILIES}
    producer = study.CaseReplay(index_path, item, [], layouts, stats, {}, output)
    for b in audit.BUDGETS:
        producer.get_stage(b)
    controls = producer.controls()
    old_selection = {
        "choices": [{"target_participant": "p", "policy_id": "p0"}],
        "policies": audit.previous.grids()[1],
    }
    controls["v5_adaptive"], old_evidence = study.old_comparator(
        index_path, item, [], layouts, stats, {}, output, old_selection
    )
    control_rows = []
    for name, fitted in controls.items():
        b = (
            old_evidence["decision"]["budget_seconds"]
            if name == "v5_adaptive"
            else int(name[5:])
            if name.startswith("daily")
            else 0
        )
        action = (
            old_evidence["decision"]["action"]
            if name == "v5_adaptive"
            else "abstain"
            if name == "zero"
            else "accept"
        )
        control_rows.append(
            study.save_score(
                output, name, fitted, case, study.row_metadata(item, name, b, action, b)
            )
        )
    lookup = {(r["session_id"], r["method"]): r for r in control_rows}
    curves = {"current_session_id": "p9", "current_post_implant_day": 9, "past_learning_curves": []}
    confirmed = []
    for cfg in policy.policy_grid():
        decision = policy.decide(curves, producer.get_stage, cfg)
        row = study.row_metadata(
            item,
            "confirmed",
            decision["requested_label_seconds"],
            decision["action"],
            decision["training_seconds"],
        )
        row.update(
            decision=decision,
            policy_id=cfg["policy_id"],
            zero_mse=lookup["p9", "zero"]["metrics"]["summed_mse"] / 2,
            daily240_r2=lookup["p9", "daily240"]["metrics"]["r2"],
        )
        confirmed.append(
            study.save_score(output, cfg["policy_id"], producer.decoder_for(decision), case, row)
        )
    evidence = {
        "participant": "p",
        "session_id": "p9",
        "stages": list(producer.stages.values()),
        "history_evidence": curves,
        "v5_comparator": old_evidence,
        "zero_model_path": lookup["p9", "zero"]["model_path"],
    }
    inputs = SimpleNamespace(arrays=lambda _item: (features, labels))
    verifier = object.__new__(audit.Verifier)
    verifier.inputs, verifier.root, verifier.items = inputs, tmp_path, []
    verifier.layouts, verifier.stats = layouts, stats
    verifier.errors, verifier.residuals = [], []
    verifier.models_checked = verifier.traces_checked = verifier.fits_checked = 0
    audit.check_case(
        verifier,
        inputs,
        item,
        evidence,
        producer.saved_fits,
        confirmed,
        control_rows,
        lookup,
        {},
        old_selection,
        True,
    )
    assert verifier.traces_checked == 10
    assert verifier.fits_checked == len(producer.saved_fits) + len(old_evidence["fits"])


def aggregate_rows():
    rows = []
    for person in ("T2", "T5", "T6", "T7", "T8", "T9", "T10", "T11"):
        for day in range(3):
            for method in audit.METHODS:
                unavailable = person == "T6" and day == 0
                r2 = (
                    -0.02
                    if method == "zero"
                    else 0.3
                    if method in ("confirmed", "daily240", "daily360")
                    else 0.2
                )
                if person == "T2" and day == 0 and method == "v5_adaptive":
                    r2 = -0.1
                rows.append(
                    {
                        "participant": person,
                        "session_id": f"{person}-{day}",
                        "method": method,
                        "action": "abstain" if method == "zero" else "accept",
                        "label_seconds": 420 if unavailable else 300,
                        "available_label_seconds": 200 if unavailable else 300,
                        "training_label_seconds": 240,
                        "eval_mask_sha256": "a" * 64,
                        "metrics": {
                            "r2": None if unavailable else r2,
                            "summed_mse": None if unavailable else 1 - r2,
                            "angular_error_degrees": None if unavailable else 60.0,
                        },
                    }
                )
    return rows


def test_independent_summary_pairs_people_and_keeps_missing_costs():
    from neural_experiments.v6 import metrics

    rows = aggregate_rows()
    actual, expected = metrics.summarize(rows), audit.summary_reference(rows)
    audit.compare_structure(actual, expected)
    assert expected["methods"]["confirmed"]["scored_sessions"] == 23
    assert expected["methods"]["confirmed"]["mean_requested_label_seconds"] == 305
    assert expected["gates"]["T2_recovered_without_harmful_acceptance"]
    assert expected["later_label_opportunities"]["capture_fraction"] == 1


def test_summary_cannot_silently_remove_unavailable_method_or_change_mask():
    rows = aggregate_rows()
    with pytest.raises(ValueError, match="inventory"):
        audit.summary_reference(rows[1:])
    rows[0]["eval_mask_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="paired"):
        audit.summary_reference(rows)


def provenance_fixture(tmp_path):
    prior = []
    seen = []
    for rank in range(1, 19):
        name = f"decoding/T2/T2_day_{rank}_decoding.mat"
        seen.append({"regular_file": True, "name": name, "size": 3, "ordinal": rank})
        member = {
            "participant_id": "T2",
            "session_id": f"T2_day_{rank}_decoding",
            "post_implant_day": rank,
            "regular_session_rank": rank,
            "member_name": name,
            "member_size": 3,
            "member_sha256": audit.sha256(__file__),
            "archive_ordinal": rank,
        }
        prior.append(member)
    for row in prior[15:]:
        row["raw_path"] = f"raw/T2/{row['session_id']}.mat"
        path = tmp_path / row["raw_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"raw")
        row["member_sha256"] = audit.sha256(path)
    receipt = {
        "status": "complete",
        "participant_id": "T2",
        "plan_sha256": audit.ACQUISITION_SHA,
        "selection": {
            "regular_session_ranks_one_based": [16, 17, 18],
            "scores_or_duration_used_for_selection": False,
        },
        "v5_development_index_sha256": "devpin",
        "v5_new_index_sha256": "newpin",
        "archive": {"file_id": 1, "provider_sha256": "a" * 64},
        "full_archive_sha256_verified": False,
        "seen_members": seen,
        "verified_prior_members": prior[:15],
        "selected": prior[15:],
        "tar_end_observed": False,
        "missing_requested_ranks": [],
    }
    path = tmp_path / "acquisition/T2.json"
    path.parent.mkdir()
    path.write_text(json.dumps(receipt))
    extra = {
        "archive_file_id": 1,
        "archive_sha256_published": "a" * 64,
        "full_archive_sha256_verified": False,
    }
    devitems = {
        r["session_id"]: {"participant_id": "T2", "source": {**r, **extra}} for r in prior[:15]
    }
    items = {
        r["session_id"]: {
            "source": {**r, **extra},
            "post_implant_day": r["post_implant_day"],
            "n_bins": 2000,
            "full_prepared_n_bins": 2000,
            "raw_source_root": str(tmp_path),
        }
        for r in prior[15:]
    }
    parent = {"v5_development_index_sha256": "devpin", "v5_new_index_sha256": "newpin"}
    index = {
        **parent,
        "participants": ["T2"],
        "acquisition_sources": [{"path": "acquisition/T2.json", "sha256": audit.sha256(path)}],
        "participant_dispositions": [
            {"participant_id": "T2", "selected_sessions": 3, "missing_requested_ranks": []}
        ],
    }
    return (
        SimpleNamespace(path=tmp_path / "index.json", index=index, items=items),
        SimpleNamespace(index=parent, items=devitems),
        {"new_session_participants": ["T2"]},
    )


def test_exact_rank_provenance_and_dispositions_are_bound(tmp_path):
    inputs, dev, plan = provenance_fixture(tmp_path)
    audit.validate_fresh_provenance(inputs, dev, plan)
    inputs.index["participant_dispositions"][0]["selected_sessions"] = 2
    with pytest.raises((ValueError, AssertionError), match="accounting"):
        audit.validate_fresh_provenance(inputs, dev, plan)


def test_rehashed_current_member_cannot_replace_authenticated_prior(tmp_path):
    inputs, dev, plan = provenance_fixture(tmp_path)
    dev.items[next(iter(dev.items))]["source"]["member_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prior fifteen"):
        audit.validate_fresh_provenance(inputs, dev, plan)


def test_extra_prepared_source_fields_are_not_silently_accepted(tmp_path):
    inputs, dev, plan = provenance_fixture(tmp_path)
    inputs.items[next(iter(inputs.items))]["source"]["unbound"] = True
    with pytest.raises((ValueError, AssertionError), match="receipt-bound"):
        audit.validate_fresh_provenance(inputs, dev, plan)
