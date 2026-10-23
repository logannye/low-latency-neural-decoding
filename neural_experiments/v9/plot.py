"""Render aggregate-only V9 evidence; no sample traces or model artifacts are read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(study_path, output):
    study = json.loads(Path(study_path).read_text())
    if study["study"] != "V9" or study["phase"] != "assessment":
        raise ValueError("published fresh V9 evidence required")
    rows = study["rows"]
    summary = study["summary"]
    primary = summary["primary_method"]
    methods = ["baseline", "duration", primary]
    labels = ["2-minute baseline", "6-minute baseline", "Trial-balanced policy"]
    colors = ["#76859A", "#D89736", "#087F8C"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    fig.patch.set_facecolor("#FAFBFD")
    for ax in axes:
        ax.set_facecolor("#FAFBFD")
        ax.grid(axis="y", color="#E1E7EE", linewidth=0.7)
        ax.set_axisbelow(True)
    cases = sorted(
        [r for r in rows if r["participant"] == "T2" and r["method"] == primary],
        key=lambda r: r["post_implant_day"],
    )
    for method, label, color in zip(methods, labels, colors, strict=True):
        by_id = {r["session_id"]: r for r in rows if r["method"] == method}
        values = [by_id[r["session_id"]]["metrics"]["r2"] for r in cases]
        axes[0].plot(range(len(cases)), values, marker="o", color=color, label=label, linewidth=2)
    axes[0].axhline(0, color="#182C45", linewidth=0.7)
    axes[0].set_xticks(range(len(cases)), [r["post_implant_day"] for r in cases])
    axes[0].set(
        xlabel="T2 post-implant day",
        ylabel="Held-out continuous R²",
        title="Every fresh T2 session",
    )
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    people = [p["participant"] for p in summary["methods"][primary]["participants"]]
    x = np.arange(len(people))
    for j, (method, color) in enumerate(zip(methods, colors, strict=True)):
        values = {
            u["participant"]: u["mean_r2"] for u in summary["methods"][method]["participants"]
        }
        axes[1].bar(x + (j - 1) * 0.24, [values[p] for p in people], width=0.23, color=color)
    axes[1].axhline(0, color="#182C45", linewidth=0.7)
    axes[1].set_xticks(x, people)
    axes[1].set(
        ylabel="Mean session R²",
        xlabel="Known participant; separate session fits",
        title="Recipe portability with calibration",
    )
    costs = [
        [
            r["calibration"]["observed_label_seconds"]
            for r in rows
            if r["method"] == primary and r["participant"] == p
        ]
        for p in people
    ]
    for j, values in enumerate(costs):
        offsets = np.linspace(-0.13, 0.13, len(values))
        axes[2].scatter(j + offsets, values, s=33, color="#087F8C", alpha=0.85)
        axes[2].plot([j - 0.2, j + 0.2], [np.median(values)] * 2, color="#182C45", linewidth=2)
    axes[2].axhline(120, color=colors[0], linestyle="--", linewidth=1)
    axes[2].axhline(360, color=colors[1], linestyle="--", linewidth=1)
    axes[2].set_xticks(x, people)
    axes[2].set(
        ylabel="Recorded seconds of labels",
        xlabel="Dots: sessions; bars: medians",
        title="Calibration cost, including timeouts",
        ylim=(0, 385),
    )
    fig.suptitle(
        "Does trial-balanced calibration rescue T2?",
        x=0.06,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#182C45",
    )
    fig.text(
        0.06,
        0.025,
        "24 fixed fresh sessions • same held-out suffix for every method • "
        "geometric direction proxy • offline evidence",
        color="#52637B",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.015, 0.065, 1, 0.91), w_pad=2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.study, args.output)
