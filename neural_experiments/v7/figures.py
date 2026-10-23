"""Render aggregate-only V7 recoverability figures from the published receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = (
    "frozen120",
    "expanding",
    "rolling60",
    "rolling120",
    "rolling240",
    "exponential60",
    "rolling120_local",
)
LABELS = (
    "Frozen startup",
    "Expanding history",
    "Recent 60 s",
    "Recent 120 s",
    "Recent 240 s",
    "60 s half-life",
    "Recent 120 s + normalization",
)


def render(source, output):
    data = json.loads(Path(source).read_text())
    summary = data["summary"]
    primary = summary["primary_T2"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1, 1.2]})
    colors = ["#66788A" if m == "frozen120" else "#168A80" for m in ORDER]
    y = np.arange(len(ORDER))
    values = np.array([primary[m]["mean_r2"] for m in ORDER])
    intervals = np.array([primary[m]["session_bootstrap_95"] for m in ORDER])
    axes[0].barh(y, values, color=colors, height=0.65)
    axes[0].errorbar(
        values,
        y,
        xerr=np.stack((values - intervals[:, 0], intervals[:, 1] - values)),
        fmt="none",
        ecolor="#253645",
        capsize=3,
        linewidth=1,
    )
    axes[0].set_yticks(y, LABELS)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#777777", linewidth=0.7)
    axes[0].set_xlabel(
        "Mean R², six recent T2 sessions\nDescriptive session-bootstrap 95% interval"
    )
    axes[0].set_title(
        "Does fresh feedback restore T2 decoding?", loc="left", fontweight="bold", pad=15
    )
    days = [r["post_implant_day"] for r in primary["frozen120"]["sessions"]]
    matrix = np.array([[r["r2"] for r in primary[m]["sessions"]] for m in ORDER])
    bound = max(float(np.abs(matrix).max()), 0.1)
    axes[1].imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    axes[1].set_xticks(np.arange(len(days)), [str(d) for d in days])
    axes[1].set_yticks(y, [])
    for i in range(len(ORDER)):
        for j in range(len(days)):
            axes[1].text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if abs(matrix[i, j]) > bound * 0.6 else "#172738",
                fontsize=10,
            )
    axes[1].set_xlabel("T2 acquisition day")
    axes[1].set_title("Every primary session is retained", loc="left", fontweight="bold", pad=15)
    disposition = "PASS" if summary["gate"]["passed"] else "FAIL"
    fig.suptitle(
        f"V7 · Exposed-data recovery gate: {disposition}",
        x=0.025,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.025,
        "Dense-label diagnostic · past-only fitting · 2 s embargo · "
        "shared future windows · not fresh validation",
        fontsize=10,
        color="#495664",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93), w_pad=3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix("." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
