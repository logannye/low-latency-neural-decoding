"""Finite prospective calibration rules; no held-out outcome enters a decision.

The callback supplies only the requested label prefix. The policy stops invoking
it as soon as a candidate is accepted. At the cap, failure emits an explicit
zero-output abstention. Group agreement is a heuristic, not a confidence bound.
"""

from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np

WARMUP = 1500
BUDGETS = (30, 60, 120, 240)
CANDIDATES = tuple(
    f"{family}_{head}"
    for family in ("tx45", "tx45_sbp")
    for head in ("prior10", "scratch1", "scratch10_zero")
)
PRIORS = ("tx45_prior10", "tx45_sbp_prior10")


def policy_grid():
    return [
        {"policy_id": f"p{i}", "history_mean_mse_ceiling": ceiling, "validation_margin": margin}
        for i, (ceiling, margin) in enumerate(itertools.product((0.40, 0.45), (0.025, 0.05)))
    ]


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _loss(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and 0 <= value <= 2.000001
    )


def _mask_hash(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=bool).tobytes()).hexdigest()


def _sectors(y, mask):
    selected = np.asarray(y[mask], dtype=float)
    angle = np.arctan2(selected[:, 1], selected[:, 0])
    sectors = np.floor((angle + np.pi) * (8 / (2 * np.pi))).astype(int) % 8
    return np.bincount(sectors, minlength=8).tolist()


def _ranges(segments, stop):
    ranges = [(s["start"], s["end"]) if isinstance(s, dict) else tuple(s) for s in segments]
    if (
        not ranges
        or ranges[0][0] != 0
        or ranges[-1][1] < stop
        or any(not _integer(a) or not _integer(b) or a < 0 or b < a for a, b in ranges)
        or any(left[1] != right[0] for left, right in zip(ranges, ranges[1:]))
    ):
        raise ValueError("Acquisition segments must partition the requested prefix")
    return [(int(a), min(int(b), stop)) for a, b in ranges if a < stop and a < b]


def stage_evidence(y, mask, segments, train_stop, validation_start, stop, predictions_by_candidate):
    """Aggregate only requested-prefix held-out labels into finite policy evidence.

    Inputs must end exactly at stop. Burn applies to both fitting and validation.
    Fixed 100-bin groups are anchored at validation_start; groups crossing an
    acquisition boundary, incomplete groups, and groups with <25 valid rows are
    excluded. Coverage and pooled loss use the identical retained validation rows.
    """
    if not _integer(stop) or stop not in tuple(WARMUP + b * 50 for b in BUDGETS):
        raise ValueError("Unplanned requested label stop")
    if train_stop != WARMUP + (stop - WARMUP) // 2 or validation_start != train_stop + 100:
        raise ValueError("Training/validation boundaries differ from the locked half-prefix rule")
    y, mask = np.asarray(y), np.asarray(mask)
    if (
        y.shape != (stop, 2)
        or mask.shape != (stop,)
        or mask.dtype != bool
        or not np.isfinite(y[mask]).all()
        or not np.allclose(np.linalg.norm(y[mask], axis=1), 1, atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("Supply only the requested finite unit-direction label prefix")
    if set(predictions_by_candidate) != set(CANDIDATES):
        raise ValueError("Every prespecified candidate prediction is required")
    predictions = {k: np.asarray(predictions_by_candidate[k], dtype=float) for k in CANDIDATES}
    if any(
        p.shape != (stop, 2)
        or not np.isfinite(p).all()
        or np.any(np.linalg.norm(p, axis=1) > 1.000001)
        for p in predictions.values()
    ):
        raise ValueError("Candidate predictions must be finite unit-disk requested prefixes")
    ranges = _ranges(segments, stop)
    burned = mask.copy()
    for start, end in ranges:
        burned[start : min(start + 62, end)] = False
    training = burned.copy()
    training[:WARMUP] = False
    training[train_stop:] = False
    eligible = burned.copy()
    eligible[:validation_start] = False
    used = np.zeros(stop, bool)
    mean = np.asarray(y[training], float).mean(0) if training.sum() >= 50 else np.zeros(2)
    groups = []
    for start in range(validation_start, stop - 99, 100):
        end = start + 100
        if not any(a <= start and end <= b for a, b in ranges):
            continue
        local = eligible[start:end]
        n = int(local.sum())
        if n < 25:
            continue
        used[start:end] = local
        observed = np.asarray(y[start:end][local], dtype=float)
        groups.append(
            {
                "start": start,
                "stop": end,
                "n": n,
                "zero_mse": float(np.square(observed).mean()),
                "prefix_mean_mse": float(np.square(observed - mean).mean()),
                "candidate_mse": {
                    k: float(np.square(observed - p[start:end][local]).mean())
                    for k, p in predictions.items()
                },
            }
        )
    return {
        "budget_seconds": (stop - WARMUP) // 50,
        "train_stop": train_stop,
        "validation_start": validation_start,
        "stop": stop,
        "training_rows": int(training.sum()),
        "training_sector_counts": _sectors(y, training),
        "validation_eligible_rows": int(eligible.sum()),
        "validation_rows": int(used.sum()),
        "validation_sector_counts": _sectors(y, used),
        "training_mask_sha256": _mask_hash(training),
        "validation_mask_sha256": _mask_hash(used),
        "groups": groups,
    }


def _validate_history(evidence):
    if set(evidence) != {
        "current_session_id",
        "current_post_implant_day",
        "warmup_available",
        "prior_candidates",
    }:
        raise ValueError("History evidence must not contain current/future target outcomes")
    if (
        not isinstance(evidence["current_session_id"], str)
        or not _integer(evidence["current_post_implant_day"])
        or not isinstance(evidence["warmup_available"], bool)
    ):
        raise ValueError("Invalid query identity/warmup geometry")
    candidates = evidence["prior_candidates"]
    if len(candidates) != 2 or [c["candidate_id"] for c in candidates] != list(PRIORS):
        raise ValueError("Prior candidates must preserve the fixed family order")
    for candidate in candidates:
        if set(candidate) != {"candidate_id", "prior_available", "probes"} or not isinstance(
            candidate["prior_available"], bool
        ):
            raise ValueError("Invalid prior readiness evidence")
        probes = candidate["probes"]
        if len(probes) > 3 or len({p["session_id"] for p in probes}) != len(probes):
            raise ValueError("At most three unique historical probes are permitted")
        previous = None
        for probe in probes:
            if (
                set(probe) != {"session_id", "post_implant_day", "n", "mse"}
                or not isinstance(probe["session_id"], str)
                or not _integer(probe["post_implant_day"])
                or probe["session_id"] == evidence["current_session_id"]
                or probe["post_implant_day"] >= evidence["current_post_implant_day"]
                or not _integer(probe["n"])
                or probe["n"] < 50
                or not _loss(probe["mse"])
            ):
                raise ValueError(
                    "Reliability probes must be eligible, strictly earlier held-out pseudoqueries"
                )
            key = (probe["post_implant_day"], probe["session_id"])
            if previous is not None and key <= previous:
                raise ValueError("Historical reliability probes are not chronological")
            previous = key


def _validate_stage(stage, budget):
    required = {
        "budget_seconds",
        "train_stop",
        "validation_start",
        "stop",
        "training_rows",
        "training_sector_counts",
        "validation_eligible_rows",
        "validation_rows",
        "validation_sector_counts",
        "training_mask_sha256",
        "validation_mask_sha256",
        "groups",
    }
    if set(stage) != required or stage["budget_seconds"] != budget:
        raise ValueError("Unexpected stage fields or requested budget; future metrics forbidden")
    stop, cut = WARMUP + budget * 50, WARMUP + budget * 25
    if (
        stage["stop"] != stop
        or stage["train_stop"] != cut
        or stage["validation_start"] != cut + 100
    ):
        raise ValueError("Stage timing is not the requested half-prefix validation")
    for key, nkey in (
        ("training_sector_counts", "training_rows"),
        ("validation_sector_counts", "validation_rows"),
    ):
        values = stage[key]
        if (
            not _integer(stage[nkey])
            or stage[nkey] < 0
            or len(values) != 8
            or any(not _integer(v) or v < 0 for v in values)
            or sum(values) != stage[nkey]
        ):
            raise ValueError("Directional coverage counts differ from label rows")
    if (
        not _integer(stage["validation_eligible_rows"])
        or stage["validation_eligible_rows"] < stage["validation_rows"]
    ):
        raise ValueError("Retained validation rows exceed observed eligibility")
    previous_stop, total = cut + 100, 0
    for group in stage["groups"]:
        if (
            set(group) != {"start", "stop", "n", "zero_mse", "prefix_mean_mse", "candidate_mse"}
            or not _integer(group["start"])
            or not _integer(group["stop"])
            or group["start"] < previous_stop
            or group["stop"] - group["start"] != 100
            or (group["start"] - (cut + 100)) % 100
            or group["stop"] > stop
            or not _integer(group["n"])
            or not 25 <= group["n"] <= 100
            or not _loss(group["zero_mse"])
            or not _loss(group["prefix_mean_mse"])
            or set(group["candidate_mse"]) != set(CANDIDATES)
            or any(not _loss(v) for v in group["candidate_mse"].values())
        ):
            raise ValueError("Invalid held-out validation group")
        previous_stop, total = group["stop"], total + group["n"]
    if total != stage["validation_rows"]:
        raise ValueError("Pooled validation row count differs from its groups")


def _stage_decision(stage, config):
    groups = stage["groups"]
    geometry = (
        stage["training_rows"] >= 100
        and stage["validation_rows"] >= 100
        and sum(v >= 10 for v in stage["training_sector_counts"]) >= 4
        and sum(v >= 10 for v in stage["validation_sector_counts"]) >= 4
        and len(groups) >= 3
    )
    checks = []
    total = stage["validation_rows"]
    null = {
        key: sum(g["n"] * g[key] for g in groups) / total if total else None
        for key in ("zero_mse", "prefix_mean_mse")
    }
    for candidate in CANDIDATES:
        loss = (
            sum(g["n"] * g["candidate_mse"][candidate] for g in groups) / total if total else None
        )
        wins = sum(
            g["candidate_mse"][candidate] < min(g["zero_mse"], g["prefix_mean_mse"]) for g in groups
        )
        passed = bool(
            geometry
            and loss <= min(null.values()) - config["validation_margin"]
            and wins >= math.ceil(2 * len(groups) / 3)
        )
        checks.append(
            {"candidate_id": candidate, "mse": loss, "winning_groups": wins, "passed": passed}
        )
    passing = [c for c in checks if c["passed"]]
    chosen = min(passing, key=lambda c: c["mse"])["candidate_id"] if passing else None
    event = {
        "budget_seconds": stage["budget_seconds"],
        "geometry_passed": bool(geometry),
        "training_rows": stage["training_rows"],
        "validation_rows": total,
        "training_sector_counts": stage["training_sector_counts"],
        "validation_sector_counts": stage["validation_sector_counts"],
        "validation_groups": len(groups),
        "null_mse": null,
        "candidates": checks,
        "selected_candidate": chosen,
    }
    return chosen, event


def decide(history_evidence, get_stage, config):
    """Lazily request only the next authorized prefix; never see final test metrics."""
    if config not in policy_grid():
        raise ValueError("Unplanned policy thresholds")
    _validate_history(history_evidence)
    events, passing = [], []
    for candidate in history_evidence["prior_candidates"]:
        values = [p["mse"] for p in candidate["probes"]]
        mean, worst = (float(np.mean(values)), max(values)) if values else (None, None)
        good = bool(
            history_evidence["warmup_available"]
            and candidate["prior_available"]
            and len(values) >= 2
            and mean <= config["history_mean_mse_ceiling"]
            and worst <= 0.50
        )
        events.append(
            {
                "budget_seconds": 0,
                "candidate_id": candidate["candidate_id"],
                "n_probes": len(values),
                "mean_mse": mean,
                "worst_mse": worst,
                "passed": good,
            }
        )
        if good:
            passing.append((mean, candidate["candidate_id"]))

    def result(action, candidate, budget, reason, available=None):
        return {
            "action": action,
            "candidate_id": candidate,
            "budget_seconds": budget,
            "requested_label_seconds": budget,
            "available_label_seconds": budget if available is None else available,
            "decision_time_seconds": 30 + (budget if available is None else available),
            "reason": reason,
            "events": events,
        }

    if passing:
        return result(
            "accept",
            min(passing, key=lambda value: value[0])[1],
            0,
            "historical_reliability_passed",
        )
    if not history_evidence["warmup_available"]:
        return result("abstain", "zero", 0, "unavailable_30s_warmup", 0)
    for budget in BUDGETS:
        stage = get_stage(budget)
        if "unavailable" in stage:
            if (
                set(stage) - {"budget_seconds", "unavailable", "available_label_seconds"}
                or stage.get("budget_seconds") != budget
                or not isinstance(stage["unavailable"], str)
            ):
                raise ValueError("Invalid unavailable-checkpoint evidence")
            available = stage.get("available_label_seconds", 0)
            if (
                not isinstance(available, (int, float))
                or not np.isfinite(available)
                or not 0 <= available <= budget
            ):
                raise ValueError("Invalid actually available label duration")
            events.append(dict(stage))
            return result("abstain", "zero", budget, "checkpoint_unavailable", available)
        _validate_stage(stage, budget)
        chosen, event = _stage_decision(stage, config)
        events.append(event)
        if chosen is not None:
            return result("accept", chosen, budget, "held_out_prefix_evidence_passed")
    return result("abstain", "zero", 240, "no_candidate_passed_at_cap")
