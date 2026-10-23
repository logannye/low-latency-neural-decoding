"""Aggregate-only figures for the locked fresh-session continual experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ("frozen120", "fixed240", "periodic240", "triggered240")
LABELS = ("Startup 120 s", "Startup 240 s", "Periodic 240 s", "Triggered 240 s")
COLORS = ("#657786", "#A8B6C0", "#087F8C", "#C56B32")


def render(source, output):
    data = json.loads(Path(source).read_text())
    if data["phase"] != "assessment" or data["study"] != "V7B":
        raise ValueError("fresh V7B aggregate required")
    summary = data["summary"]["methods"]
    people = [u["participant"] for u in summary["periodic240"]["participants"]]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
        }
    )
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13.3, 6.1), gridspec_kw={"width_ratios": [1.0, 1.4]}
    )
    for i, (method, label, color) in enumerate(zip(ORDER, LABELS, COLORS, strict=True)):
        values = summary[method]
        mean = values["mean_r2"]
        low, high = values["participant_bootstrap_95"]
        left.barh(i, mean, color=color, height=0.55)
        left.errorbar(
            mean,
            i,
            xerr=[[mean - low], [high - mean]],
            fmt="none",
            color="#233341",
            capsize=4,
            linewidth=1.2,
        )
        left.text(
            0.99,
            i + 0.30,
            f"R² {mean:.3f}  ·  labels {values['mean_available_label_seconds']:.1f} s"
            f"  ·  interruption {values['mean_interruption_seconds']:.1f} s",
            transform=left.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.3,
            color="#44515D",
        )
        per_person = {u["participant"]: u["mean_r2"] for u in values["participants"]}
        right.plot(
            range(len(people)),
            [per_person[p] for p in people],
            marker="o",
            markersize=5,
            linewidth=1.3,
            color=color,
            label=label,
        )
    left.set_yticks(range(4), LABELS)
    left.invert_yaxis()
    left.set_title("Whole-session accuracy and cost", pad=17)
    left.set_xlabel("Equal-participant mean R²\nParticipant-bootstrap 95% interval")
    left.axvline(0, color="#99A2A9", linewidth=0.8)
    right.set_xticks(range(len(people)), people)
    right.set_title("Every participant, including the failures", pad=17)
    right.set_ylabel("Mean session R²")
    right.set_xlabel("Known participant, fresh sessions")
    right.axhline(0, color="#99A2A9", linewidth=0.8)
    right.grid(axis="y", alpha=0.16)
    right.legend(loc="best", frameon=False, ncol=2, fontsize=9)
    n = len({r["session_id"] for r in data["rows"]})
    scored = summary["periodic240"]["scored_sessions"]
    fig.suptitle(
        f"Continual recalibration · {scored}/{n} fresh sessions scored",
        x=0.025,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.055,
        "Common 152–600 s interval, truncated at recording end. "
        "Calibration and update delays stay in the score.",
        fontsize=10,
        color="#44515D",
    )
    fig.text(
        0.025,
        0.022,
        "Costs include all selected recordings. Periodic and triggered label budgets match "
        "on complete 600 s recordings; short cases can differ.",
        fontsize=9,
        color="#44515D",
    )
    fig.tight_layout(rect=(0, 0.115, 1, 0.92), w_pad=2.6)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"v7-continual.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    t2 = [r for r in data["rows"] if r["participant"] == "T2"]
    ids = sorted({(r["post_implant_day"], r["session_id"]) for r in t2})
    lookup = {(r["session_id"], r["method"]): r for r in t2}
    fig, axes = plt.subplots(1, len(ids), figsize=(13.3, 4.8), sharey=True, squeeze=False)
    for ax, (day, sid) in zip(axes[0], ids, strict=True):
        for method, label, color in zip(ORDER, LABELS, COLORS, strict=True):
            row = lookup[sid, method]
            windows = row["diagnostic_windows"]
            zero_windows = lookup[sid, "zero"]["diagnostic_windows"]
            ax.plot(
                [w["stop_bin"] * 0.02 for w in windows],
                [
                    z["metrics"]["summed_mse"] - w["metrics"]["summed_mse"]
                    if w["metrics"] is not None and w["metrics"]["summed_mse"] is not None
                    else np.nan
                    for w, z in zip(windows, zero_windows, strict=True)
                ],
                color=color,
                linewidth=1.3,
                marker=".",
                markersize=3,
                label=label,
            )
        ax.axhline(0, color="#99A2A9", linewidth=0.8)
        ax.set_title(f"T2 day {day}")
        ax.set_xlabel("Window end, recorded seconds")
        ax.grid(axis="y", alpha=0.16)
    axes[0, 0].set_ylabel("MSE improvement over zero output\n(30-second windows; higher is better)")
    axes[0, -1].legend(frameon=False, fontsize=8, ncol=2, loc="best")
    fig.suptitle(
        "T2 · sustained performance across the entire deployment interval",
        x=0.025,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.025,
        "Scorer-only diagnostics. Zero-output calibration windows show no improvement; "
        "gaps represent insufficient valid observations.",
        fontsize=9,
        color="#44515D",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.92), w_pad=2)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"v7-t2-continual.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
