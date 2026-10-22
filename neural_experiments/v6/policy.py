"""Fixed-artifact confirmation and whole-person-excluded V6 policy ranking.

Confirmation never ranks alternative models. Inner-selection evidence is retained
for the exposed diagnostic only; prospective nonzero stages use one fixed recipe.
"""

from __future__ import annotations

import hashlib

import numpy as np

CANDIDATES = tuple(
    f"{family}_{head}_{normalization}"
    for family in ("tx45", "tx45_sbp")
    for head in ("prior10", "scratch1", "scratch10_zero")
    for normalization in ("first30", "fit_available")
)
FIXED_CANDIDATE = "tx45_sbp_scratch1_fit_available"
BUDGETS = (0, 60, 120, 240, 360)


def policy_grid():
    return [
        {"policy_id": f"c{i * 2 + j}", "gain_margin": margin, "pooled_r2_min": minimum}
        for i, margin in enumerate((0.0, 0.025))
        for j, minimum in enumerate((0.0, 0.05))
    ]


def _hash(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=bool).tobytes()).hexdigest()


def _sectors(y, mask):
    selected = np.asarray(y[mask], dtype=np.float64)
    angle = np.arctan2(selected[:, 1], selected[:, 0])
    bins = np.floor((angle + np.pi) * (8 / (2 * np.pi))).astype(int) % 8
    return np.bincount(bins, minlength=8).tolist()


def _prefix(y, mask, segments, stop):
    y, mask = np.asarray(y), np.asarray(mask)
    if (
        type(stop) is not int
        or stop <= 0
        or y.shape != (stop, 2)
        or mask.shape != (stop,)
        or mask.dtype != bool
        or not np.isfinite(y[mask]).all()
        or not np.allclose(np.linalg.norm(y[mask], axis=1), 1, atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("Only the requested finite unit-direction label prefix is permitted")
    ranges = [(s["start"], s["end"]) if isinstance(s, dict) else tuple(s) for s in segments]
    if (
        not ranges
        or ranges[0][0] != 0
        or ranges[-1][1] < stop
        or any(type(a) is not int or type(b) is not int or a < 0 or b < a for a, b in ranges)
        or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:]))
    ):
        raise ValueError("Acquisition segments must partition the prefix")
    ranges = [(a, min(b, stop)) for a, b in ranges if a < stop and a < b]
    burned = mask.copy()
    for a, b in ranges:
        burned[a : min(a + 62, b)] = False
    return np.asarray(y, dtype=np.float64), burned, ranges


def _prediction(prediction, stop):
    prediction = np.asarray(prediction, dtype=np.float64)
    if (
        prediction.shape != (stop, 2)
        or not np.isfinite(prediction).all()
        or np.any(np.linalg.norm(prediction, axis=1) > 1.000001)
    ):
        raise ValueError(
            "Only the frozen artifact's finite unit-disk prefix prediction is permitted"
        )
    return prediction


def inner_selection(y, mask, segments, train_stop, validation_start, stop, predictions):
    """Pick one candidate on earlier inner labels, before its full fit and confirmation."""
    if (
        type(train_stop) is not int
        or type(validation_start) is not int
        or train_stop <= 1500
        or validation_start != train_stop + 100
        or train_stop != 1500 + (stop - 1500) // 2
    ):
        raise ValueError("Inner split must use half the prospective training interval and2s purge")
    if set(predictions) != set(CANDIDATES):
        raise ValueError("Exactly the twelve prespecified candidate predictions are required")
    y, burned, ranges = _prefix(y, mask, segments, stop)
    predictions = {k: _prediction(predictions[k], stop) for k in CANDIDATES}
    train = burned.copy()
    train[:1500], train[train_stop:] = False, False
    selected = np.zeros(stop, bool)
    groups = 0
    for a in range(validation_start, stop - 99, 100):
        b = a + 100
        if any(left <= a and b <= right for left, right in ranges) and burned[a:b].sum() >= 25:
            selected[a:b] = burned[a:b]
            groups += 1
    training_counts, validation_counts = _sectors(y, train), _sectors(y, selected)
    geometry = bool(
        train.sum() >= 100
        and selected.sum() >= 100
        and groups >= 3
        and sum(v >= 10 for v in training_counts) >= 4
        and sum(v >= 10 for v in validation_counts) >= 4
    )
    candidates = [
        {
            "candidate_id": cid,
            "mse": float(np.mean((y[selected] - prediction[selected]) ** 2))
            if selected.any()
            else None,
        }
        for cid, prediction in predictions.items()
    ]
    chosen = min(candidates, key=lambda c: c["mse"])["candidate_id"] if geometry else None
    return {
        "selected_candidate": chosen,
        "geometry_passed": geometry,
        "train_stop": train_stop,
        "validation_start": validation_start,
        "stop": stop,
        "training_rows": int(train.sum()),
        "training_sector_counts": training_counts,
        "validation_rows": int(selected.sum()),
        "validation_sector_counts": validation_counts,
        "validation_groups": groups,
        "training_mask_sha256": _hash(train),
        "validation_mask_sha256": _hash(selected),
        "candidates": candidates,
    }


def confirmation_evidence(y, mask, segments, start, stop, prediction, training_mean):
    """Measure one locked model on subsequent labels; no alternate models are accepted."""
    if type(start) is not int or start < 1500 or start >= stop:
        raise ValueError("Confirmation must follow the unlabeled warmup and frozen fit")
    y, burned, ranges = _prefix(y, mask, segments, stop)
    prediction = _prediction(prediction, stop)
    mean = np.asarray(training_mean, dtype=np.float64)
    if mean.shape != (2,) or not np.isfinite(mean).all() or np.linalg.norm(mean) > 1.000001:
        raise ValueError("Training-only null mean must be a finite fixed unit-disk vector")

    def window(a, b):
        chosen = burned.copy()
        chosen[:a], chosen[b:] = False, False
        n = int(chosen.sum())
        actual = y[chosen]
        mse = float(np.mean((actual - prediction[chosen]) ** 2)) if n else None
        zero = float(np.mean(actual**2)) if n else None
        baseline = float(np.mean((actual - mean) ** 2)) if n else None
        sst = float(np.sum((actual - actual.mean(0)) ** 2)) if n else 0.0
        return {
            "start_bin": a,
            "stop_bin": b,
            "n": n,
            "direction_sector_counts": _sectors(y, chosen),
            "mse": mse,
            "r2": 1 - float(np.sum((actual - prediction[chosen]) ** 2)) / sst if sst > 0 else None,
            "zero_mse": zero,
            "training_mean_mse": baseline,
            "gain_vs_zero": zero - mse if n else None,
            "gain_vs_training_mean": baseline - mse if n else None,
            "mask_sha256": _hash(chosen),
        }

    mid = (start + stop) // 2
    groups = []
    for a in range(start, stop - 99, 100):
        b = a + 100
        if any(left <= a and b <= right for left, right in ranges):
            values = window(a, b)
            if values["n"] >= 25:
                groups.append(values)
    return {
        "start_bin": start,
        "stop_bin": stop,
        "pooled": window(start, stop),
        "first_half": window(start, mid),
        "last_half": window(mid, stop),
        "groups": groups,
    }


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def budget_floor(history_evidence):
    """Use at most three already scored, strictly earlier fixed-recipe pseudoqueries."""
    current = history_evidence["current_session_id"]
    day = history_evidence["current_post_implant_day"]
    curves = history_evidence["past_learning_curves"]
    if (
        not _finite(day)
        or len(curves) > 3
        or len({c["session_id"] for c in curves}) != len(curves)
        or any(
            c["session_id"] == current
            or not _finite(c["post_implant_day"])
            or c["post_implant_day"] >= day
            or set(c["budgets"]) != {"120", "240", "360"}
            for c in curves
        )
    ):
        raise ValueError("Budget routing requires at most three distinct strictly earlier cases")
    for curve in curves:
        values = list(curve["budgets"].values())
        if len({v["n"] for v in values}) != 1 or len({v["r2"] is None for v in values}) != 1:
            raise ValueError("Past learning curves must share identical scoring availability")
        for value in values:
            if (
                type(value["n"]) is not int
                or value["n"] < 0
                or (
                    value["r2"] is not None
                    and (
                        value["n"] < 50
                        or not _finite(value["r2"])
                        or not _finite(value["coordinate_mse"])
                        or value["coordinate_mse"] < 0
                    )
                )
                or (value["r2"] is None and value["coordinate_mse"] is not None)
            ):
                raise ValueError("Malformed fixed-recipe historical score")
    paired = [c for c in curves if c["budgets"]["120"]["r2"] is not None]
    means = {
        str(b): float(np.mean([c["budgets"][str(b)]["r2"] for c in paired])) if paired else None
        for b in (120, 240, 360)
    }
    comparisons = []
    floor = 0
    for earlier, later in ((120, 240), (240, 360)):
        gains = [c["budgets"][str(later)]["r2"] - c["budgets"][str(earlier)]["r2"] for c in paired]
        gain = float(np.mean(gains)) if gains else None
        positive = sum(g > 0 for g in gains)
        triggered = bool(
            len(paired) >= 2
            and means["120"] <= 0.05
            and means[str(later)] > 0
            and gain >= 0.03
            and positive * 3 >= len(paired) * 2
        )
        comparisons.append(
            {
                "earlier_training_seconds": earlier,
                "later_training_seconds": later,
                "paired_cases": len(paired),
                "source_session_ids": [c["session_id"] for c in paired],
                "mean_120_r2": means["120"],
                "mean_later_r2": means[str(later)],
                "mean_paired_gain": gain,
                "positive_paired_gains": positive,
                "triggered": triggered,
            }
        )
        if triggered:
            floor = later
    return {
        "training_seconds": floor,
        "reason": "strict_past_later_label_benefit" if floor else "no_supported_budget_floor",
        "source_session_ids": [c["session_id"] for c in curves],
        "paired_comparisons": comparisons,
    }


def _confirmation_passed(evidence, config, start, stop):
    if evidence["start_bin"] != start or evidence["stop_bin"] != stop:
        raise ValueError("Confirmation evidence differs from the fixed temporal window")
    midpoint = (start + stop) // 2
    for key, a, b in (
        ("pooled", start, stop),
        ("first_half", start, midpoint),
        ("last_half", midpoint, stop),
    ):
        window = evidence[key]
        sectors = window["direction_sector_counts"]
        if (
            window["start_bin"] != a
            or window["stop_bin"] != b
            or type(window["n"]) is not int
            or not 0 <= window["n"] <= b - a
            or len(sectors) != 8
            or any(type(v) is not int or v < 0 for v in sectors)
            or sum(sectors) != window["n"]
        ):
            raise ValueError("Malformed confirmation geometry")
        if window["n"] < 100 or sum(v >= 10 for v in sectors) < 4:
            return False, "insufficient_confirmation_geometry"
        if any(
            not _finite(window[k])
            for k in (
                "mse",
                "zero_mse",
                "training_mean_mse",
                "gain_vs_zero",
                "gain_vs_training_mean",
            )
        ):
            raise ValueError("Malformed confirmation losses")
    threshold = config["pooled_r2_min"]
    r2 = evidence["pooled"]["r2"]
    if r2 is None or not _finite(r2) or not (r2 > 0 if threshold == 0 else r2 >= threshold):
        return False, "pooled_r2_below_threshold"
    margin = config["gain_margin"]
    if any(
        not (evidence[half][gain] > 0 if margin == 0 else evidence[half][gain] >= margin)
        for half in ("first_half", "last_half")
        for gain in ("gain_vs_zero", "gain_vs_training_mean")
    ):
        return False, "temporal_half_gain_below_threshold"
    return True, "frozen_artifact_confirmed"


def decide(history_evidence, get_stage, config):
    """Request only visited stages; never refit or inspect future scores after acceptance."""
    if config not in policy_grid():
        raise ValueError("Unplanned confirmation policy")
    floor = budget_floor(history_evidence)
    events = []

    def result(action, stage, reason):
        return {
            "action": action,
            "training_seconds": stage["training_seconds"],
            "requested_label_seconds": stage["requested_label_seconds"],
            "available_label_seconds": stage["available_label_seconds"],
            "decision_time_seconds": stage["decision_time_seconds"],
            "candidate_id": stage.get("candidate_id") if action == "accept" else None,
            "model_sha256": stage.get("model_sha256") if action == "accept" else None,
            "reason": reason,
            "events": events,
            "floor": floor,
        }

    for budget in BUDGETS:
        if budget < floor["training_seconds"]:
            continue
        stage = get_stage(budget)
        cost, decision_time = budget + 60, budget + 90
        if (
            stage["training_seconds"] != budget
            or stage["requested_label_seconds"] != cost
            or not _finite(stage["available_label_seconds"])
            or not 0 <= stage["available_label_seconds"] <= cost
            or not _finite(stage["decision_time_seconds"])
            or not 0 <= stage["decision_time_seconds"] <= decision_time
        ):
            raise ValueError("Requested confirmation stage/cost mismatch")
        if stage.get("unavailable"):
            events.append(
                {"training_seconds": budget, "accepted": False, "reason": stage["unavailable"]}
            )
            return result("abstain", stage, "incomplete_requested_confirmation")
        candidate = stage["candidate_id"]
        digest = stage["model_sha256"]
        fit_stop, start, stop = 1500 + budget * 50, 1600 + budget * 50, 4500 + budget * 50
        if (
            stage["available_label_seconds"] != cost
            or stage["decision_time_seconds"] != decision_time
            or stage["fit_stop_bin"] != fit_stop
            or stage["confirmation_start_bin"] != start
            or stage["confirmation_stop_bin"] != stop
            or candidate
            not in (("tx45_prior0", "tx45_sbp_prior0") if budget == 0 else (FIXED_CANDIDATE,))
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Confirmation must bind the exact prespecified frozen artifact")
        passed, reason = _confirmation_passed(stage["confirmation"], config, start, stop)
        events.append(
            {
                "training_seconds": budget,
                "model_sha256": digest,
                "candidate_id": candidate,
                "accepted": passed,
                "reason": reason,
            }
        )
        if passed:
            return result("accept", stage, reason)
    return result("abstain", stage, "no_confirmed_artifact_at_cap")


def choose_policy(rows, configs, people):
    """Exclude each complete target person; risk/coverage first, then accuracy and cost.

    zero_mse is coordinate-mean MSE. metrics.summed_mse is divided by two.
    Burden denominators include all scored cases; cost includes unscored cases.
    """
    ids = [c["policy_id"] for c in configs]
    if not ids or len(set(ids)) != len(ids) or len(set(people)) != len(people):
        raise ValueError("Unique nonempty policy and participant inventories required")
    lookup = {}
    inventory = {p: set() for p in people}
    for row in rows:
        person, sid, policy = row["participant"], row["session_id"], row["policy_id"]
        key = person, sid, policy
        if person not in people or policy not in ids or key in lookup:
            raise ValueError("Duplicate or unexpected source policy case")
        if (
            row["action"] not in ("accept", "abstain")
            or not np.isfinite(row["label_seconds"])
            or not 0 <= row["label_seconds"] <= 420
        ):
            raise ValueError("Invalid source action or requested label cost")
        lookup[key] = row
        inventory[person].add(sid)
    if any(not sessions for sessions in inventory.values()):
        raise ValueError("Every declared source person requires an explicit case inventory")
    for person, sessions in inventory.items():
        for sid in sessions:
            group = [lookup.get((person, sid, policy)) for policy in ids]
            if (
                any(r is None for r in group)
                or len({r["metrics"]["r2"] is None for r in group}) != 1
            ):
                raise ValueError("Policies must share identical session/scoring availability")
            if (
                len({r["daily240_r2"] for r in group}) != 1
                or len({r["zero_mse"] for r in group}) != 1
            ):
                raise ValueError("Policy comparators differ on the matched source case")
            for row in group:
                if row["metrics"]["r2"] is not None:
                    numbers = [
                        row["metrics"]["r2"],
                        row["metrics"]["summed_mse"],
                        row["daily240_r2"],
                        row["zero_mse"],
                    ]
                    if (
                        not all(isinstance(v, (int, float)) and np.isfinite(v) for v in numbers)
                        or min(numbers[1], numbers[3]) < 0
                    ):
                        raise ValueError("Invalid scored source metrics or comparator units")
    choices = []
    for target in people:
        sources = [p for p in people if p != target]
        ranking = []
        for policy in ids:
            accuracy, baseline, coverage, harmful, nonpositive, costs = [], [], [], [], [], []
            for person in sources:
                requested = [lookup[person, sid, policy] for sid in sorted(inventory[person])]
                scored = [r for r in requested if r["metrics"]["r2"] is not None]
                costs.append(float(np.mean([r["label_seconds"] for r in requested])))
                if not scored:
                    continue
                accuracy.append(float(np.mean([r["metrics"]["r2"] for r in scored])))
                baseline.append(float(np.mean([r["daily240_r2"] for r in scored])))
                coverage.append(sum(r["action"] == "accept" for r in scored) / len(scored))
                harmful.append(
                    sum(
                        r["action"] == "accept"
                        and r["metrics"]["summed_mse"] / 2 > r["zero_mse"] + 1e-12
                        for r in scored
                    )
                    / len(scored)
                )
                nonpositive.append(
                    sum(r["action"] == "accept" and r["metrics"]["r2"] <= 0 for r in scored)
                    / len(scored)
                )
            if not accuracy:
                raise ValueError("Source policy cannot be ranked without eligible source people")
            mean_r2, daily, cov, harm, fail, cost = [
                float(np.mean(v))
                for v in (accuracy, baseline, coverage, harmful, nonpositive, costs)
            ]
            ranking.append(
                {
                    "policy_id": policy,
                    "mean_r2": mean_r2,
                    "daily240_mean_r2": daily,
                    "mean_coverage": cov,
                    "mean_harmful_burden": harm,
                    "mean_nonpositive_burden": fail,
                    "mean_label_seconds": cost,
                    "eligible_source_people": len(accuracy),
                    "source_people_with_cost": len(costs),
                    "feasible": cov >= 0.6 and mean_r2 >= daily - 0.05,
                    "objective": mean_r2 - 0.10 * harm - 0.05 * fail - 0.02 * cost / 420,
                }
            )
        feasible = [r for r in ranking if r["feasible"]]
        best = (
            min(
                feasible,
                key=lambda r: (
                    r["mean_harmful_burden"],
                    r["mean_nonpositive_burden"],
                    -r["mean_r2"],
                    r["mean_label_seconds"],
                ),
            )
            if feasible
            else max(ranking, key=lambda r: r["objective"])
        )
        choices.append(
            {
                "target_participant": target,
                "policy_id": best["policy_id"],
                "source_participants": sources,
                "selection_rule": "feasible_risk_accuracy_cost"
                if feasible
                else "fallback_fixed_utility",
                "ranking": ranking,
            }
        )
    return choices
