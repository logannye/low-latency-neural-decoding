"""Standalone aggregate figures for the locked V6 held-out assessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_experiments.v5.figures import save_figure

COLORS = {
    "v5_adaptive": "#8d99ae",
    "daily120": "#c08a43",
    "daily240": "#836537",
    "daily360": "#6652a3",
    "confirmed": "#117c83",
}
LABELS = {
    "v5_adaptive": "Previous policy",
    "daily120": "Fixed 120 s",
    "daily240": "Fixed 240 s",
    "daily360": "Fixed 360 s",
    "confirmed": "Confirmed policy",
}


def render(study_path, output):
    study = json.loads(Path(study_path).read_text())
    if (
        study.get("status") != "complete"
        or study.get("independent_audit", {}).get("status") != "PASS"
    ):
        raise ValueError("Complete independently audited aggregate required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = study["summary"]["methods"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    people = sorted(
        (r["participant"] for r in summary["confirmed"]["units"]), key=lambda p: int(p[1:])
    )
    x = np.arange(len(people))
    for method, offset in (("v5_adaptive", -0.23), ("daily240", 0), ("confirmed", 0.23)):
        values = {r["participant"]: r["r2"] for r in summary[method]["units"]}
        axes[0].bar(
            x + offset,
            [values[p] for p in people],
            0.22,
            color=COLORS[method],
            label=LABELS[method],
        )
    axes[0].set_xticks(x, people)
    axes[0].axhline(0, color="#39424e", linewidth=0.8)
    axes[0].set_ylabel("Held-out R², mean within participant")
    axes[0].set_title("Every participant, including abstentions", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    for i, method in enumerate(COLORS):
        row = summary[method]
        value, (lo, hi) = row["mean_r2"], row["r2_bootstrap_95"]
        axes[1].errorbar(
            value,
            i,
            xerr=[[value - lo], [hi - value]],
            fmt="o",
            color=COLORS[method],
            capsize=4,
            markersize=7,
        )
        axes[1].text(
            1.01,
            i,
            f"{row['mean_requested_label_seconds']:g} s",
            transform=axes[1].get_yaxis_transform(),
            va="center",
            fontsize=9,
        )
    axes[1].set_yticks(range(len(COLORS)), [LABELS[m] for m in COLORS])
    axes[1].set_ylim(-0.7, len(COLORS) - 0.3)
    axes[1].axvline(0, color="#39424e", linewidth=0.8)
    axes[1].set_xlabel("Mean R² · participant-bootstrap 95% interval")
    axes[1].set_title("Accuracy and requested labels", loc="left")
    axes[1].text(1.01, 1.01, "Labels", transform=axes[1].transAxes, fontsize=9)
    fig.suptitle(
        "Calibration policy audit · fresh sessions, common 452–600 s window",
        fontsize=14,
        x=0.03,
        ha="left",
    )
    save_figure(fig, output / "v6-confirmed-calibration")
    plt.close(fig)

    rows = [r for r in study["cases"] if r["participant"] == "T2"]
    days = sorted({r["post_implant_day"] for r in rows})
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]}
    )
    for i, method in enumerate(COLORS):
        values = {r["post_implant_day"]: r["metrics"]["r2"] for r in rows if r["method"] == method}
        axes[0].bar(
            np.arange(len(days)) + (i - 2) * 0.16,
            [values[d] for d in days],
            0.15,
            label=LABELS[method],
            color=COLORS[method],
        )
    axes[0].set_xticks(np.arange(len(days)), [f"Day {d}" for d in days])
    axes[0].axhline(0, color="#39424e", linewidth=0.8)
    axes[0].set_ylabel("Held-out R²")
    axes[0].set_title("T2 · every selected session", loc="left")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for i, method in enumerate(("v5_adaptive", "confirmed")):
        row = summary[method]
        counts = [
            len(row["nonpositive_accepted_sessions"]),
            len(row["harmful_vs_zero_accepted_sessions"]),
        ]
        axes[1].bar(
            np.arange(2) + (i - 0.5) * 0.32,
            counts,
            0.30,
            color=COLORS[method],
            label=LABELS[method],
        )
        for x0, value in zip(np.arange(2) + (i - 0.5) * 0.32, counts, strict=True):
            axes[1].text(x0, value + 0.06, str(value), ha="center")
    axes[1].set_xticks([0, 1], ["Accepted R² ≤ 0", "Accepted worse\nthan zero output"])
    axes[1].set_ylabel("Number of fresh sessions · whole cohort")
    axes[1].set_title("Accepted failures, two distinct definitions", loc="left", fontsize=11)
    axes[1].set_ylim(0, max(1, axes[1].get_ylim()[1]) + 0.5)
    axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].legend(frameon=False, fontsize=8)
    save_figure(fig, output / "v6-t2-and-acceptance")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    render(a.study, a.output)


if __name__ == "__main__":
    main()
