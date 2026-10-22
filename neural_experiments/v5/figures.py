"""Standalone figures from independently audited, aggregate-only V5 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "history0": "#8d99ae",
    "history30": "#7565a8",
    "daily120": "#c98438",
    "daily240": "#8b5a27",
    "adaptive": "#177e89",
}
LABELS = {
    "history0": "History, no labels",
    "history30": "History + 30 s",
    "daily120": "Daily 120 s",
    "daily240": "Daily 240 s",
    "adaptive": "Adaptive policy",
}


def save_figure(fig, stem):
    for extension in ("png", "svg"):
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, bbox_inches="tight")
        if extension == "svg":
            lines = (line.rstrip() for line in path.read_text().splitlines())
            path.write_text("\n".join(lines) + "\n")


def render(study_path, output):
    study = json.loads(Path(study_path).read_text())
    if (
        study.get("status") != "complete"
        or study.get("independent_audit", {}).get("status") != "PASS"
    ):
        raise ValueError("Audited complete aggregate required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    summary = study["summary"]["domains"]["new_sessions"]["methods"]
    people = [r["unit"] for r in summary["adaptive"]["units"]]
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]}, layout="constrained"
    )
    x = np.arange(len(people))
    for method, offset in [("history0", -0.22), ("daily120", 0), ("adaptive", 0.22)]:
        values = {r["unit"]: r["r2"] for r in summary[method]["units"]}
        axes[0].bar(
            x + offset,
            [values[p] for p in people],
            0.21,
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[0].axhline(0, color="#39424e", linewidth=0.8)
    axes[0].set_xticks(x, people)
    axes[0].set_ylabel("Held-out R², mean within participant")
    axes[0].set_title("Fresh sessions · identical 272–600 s score window", loc="left", fontsize=12)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    for method in COLORS:
        m = summary[method]
        mean = m["mean_r2"]
        lo, hi = m["r2_bootstrap_95"]
        cost = m["mean_requested_label_seconds"]
        axes[1].errorbar(
            cost,
            mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=COLORS[method],
            capsize=4,
            markersize=7,
        )
        axes[1].annotate(
            LABELS[method],
            (cost, mean),
            xytext=(12, -40) if method == "adaptive" else (5, 7),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].axhline(0, color="#39424e", linewidth=0.8)
    axes[1].set_xlim(-12, 290)
    axes[1].set_xlabel("Requested fresh labels (s), equal-person mean")
    axes[1].set_ylabel("Mean R² and participant-bootstrap 95% interval")
    axes[1].set_title("Accuracy and calibration cost", loc="left", fontsize=12)
    fig.suptitle("Adaptive calibration under recording drift", fontsize=16, x=0.04, ha="left")
    save_figure(fig, output / "v5-adaptive-calibration")
    plt.close(fig)
    rows = [r for r in study["cases"] if r["domain"] == "new_sessions" and r["participant"] == "T2"]
    days = sorted({r["post_implant_day"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for i, method in enumerate(("zero", "history0", "daily120", "daily240", "adaptive")):
        rr = {r["post_implant_day"]: r for r in rows if r["method"] == method}
        values = [rr[d]["metrics"]["r2"] for d in days]
        ax.bar(
            np.arange(len(days)) + (i - 2) * 0.16,
            values,
            0.15,
            label=LABELS.get(method, "Zero control"),
            color=COLORS.get(method, "#bbc0c6"),
        )
    ax.axhline(0, color="#39424e", linewidth=0.8)
    ax.set_xticks(np.arange(len(days)), [f"Day {d}" for d in days])
    ax.set_ylabel("Held-out R²")
    ax.set_title("T2: every fresh session, including abstentions", loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower left")
    save_figure(fig, output / "v5-t2-transfer")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    render(a.study, a.output)


if __name__ == "__main__":
    main()
