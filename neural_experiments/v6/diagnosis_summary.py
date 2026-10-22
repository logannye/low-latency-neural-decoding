"""Derive compact exposed-pilot evidence without fitting or reading sample arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

FIXED = "tx45_sbp_scratch1_fit_available"


def summarize(path):
    path = Path(path)
    source = json.loads(path.read_text())
    if source["status"] != "complete" or len(source["fits"]) != 1440 or len(source["prior0"]) != 15:
        raise ValueError("complete approved pilot inventory required")
    fixed = {
        (r["post_implant_day"], r["budget"]): r
        for r in source["fits"]
        if r["phase"] == "full" and r["candidate_id"] == FIXED
    }
    chosen = {(r["post_implant_day"], r["budget"]): r for r in source["stages"]}
    days = sorted({day for day, _ in fixed})
    rows, histories = [], []
    for day in days:
        for budget in (60, 120, 240, 360):
            entry = {"post_implant_day": day, "budget": budget}
            for name, values in (("fixed", fixed), ("inner_selected", chosen)):
                r = values[day, budget]
                entry[name] = {
                    "candidate_id": r["candidate_id"],
                    "model_sha256": r["model_sha256"],
                    "normalization_stop_bin": r["fit"]["normalization_stop_bin"],
                    "common": r["common"],
                    "confirmation": r["confirmation"],
                }
            rows.append(entry)
        earlier = [old for old in days if old < day][-3:]
        if len(earlier) < 2:
            continue
        entry = {"post_implant_day": day, "history_days": earlier}
        for name, values in (("fixed", fixed), ("inner_selected", chosen)):
            scores = {
                str(b): [values[old, b]["common"]["r2"] for old in earlier] for b in (120, 240, 360)
            }
            entry[name] = {
                "r2_by_budget": scores,
                "mean_r2_by_budget": {b: mean(v) for b, v in scores.items()},
                "paired_mean_gain_vs120": {
                    str(b): mean(x - y for x, y in zip(scores[str(b)], scores["120"], strict=True))
                    for b in (240, 360)
                },
            }
        histories.append(entry)
    aggregates = []
    for name, condition in (
        ("all15", lambda d: True),
        ("older9", lambda d: d < 187),
        ("last6", lambda d: d >= 187),
        ("last3", lambda d: d >= 214),
    ):
        selected_days = [d for d in days if condition(d)]
        for budget in (120, 240, 360):
            scores = [fixed[d, budget]["common"]["r2"] for d in selected_days]
            aggregates.append(
                {
                    "group": name,
                    "n": len(scores),
                    "budget": budget,
                    "mean_r2": mean(scores),
                    "positive": sum(v > 0 for v in scores),
                }
            )
    return {
        "status": "complete",
        "scope": "Exploratory already-exposed <=600s T2 recordings only",
        "pilot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "summary_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256": source["plan_sha256"],
        "model_source_sha256": source["source_sha256"]["neural_experiments/v6/model.py"],
        "common_scoring_seconds": [452, 600],
        "fixed_candidate": FIXED,
        "rows": rows,
        "strictly_earlier_histories": histories,
        "fixed_aggregates": aggregates,
        "sequential": source["sequential"],
        "original_v5": source["original_v5"],
        "interpretation": (
            "Half-window R2 uses each half's own target mean and variance. "
            "Its denominator changes with directional coverage; "
            "it is not equivalent to gain versus zero. "
            "Confirmation does not guarantee later accuracy. "
            "All label/normalizer boundaries are explicit."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.pilot)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
