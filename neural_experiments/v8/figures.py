"""Publication figures from allowlisted fresh V8 aggregate results only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_experiments.v3.study import digest, write_json

COLORS = {
    "baseline": "#365D8D",
    "representation": "#D77835",
    "phase": "#8B5CA6",
    "weighting": "#3E867E",
    "zero": "#9AA3AD",
}


def generate(study_path, output):
    study_path, output = Path(study_path), Path(output)
    study = json.loads(study_path.read_text())
    if study.get("phase") != "assessment":
        raise ValueError("fresh aggregate report required")
    rows = study["rows"]
    people = sorted({r["participant"] for r in rows}, key=lambda p: int(p[1:]))
    means = {}
    for person in people:
        for method in {r["method"] for r in rows}:
            values = [
                r["metrics"]["r2"]
                for r in rows
                if r["participant"] == person
                and r["method"] == method
                and r["metrics"]["r2"] is not None
            ]
            means[person, method] = float(np.mean(values)) if values else np.nan
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FAFBFD",
            "axes.facecolor": "#FAFBFD",
        }
    )
    fig = plt.figure(figsize=(14, 8.7), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.1))
    ax = fig.add_subplot(grid[0, 0])
    methods = [
        "zero",
        "baseline",
        "selected_all_uniform",
        "baseline_all_equal_trial",
        "baseline_clean_uniform",
        "primary_onset",
    ]
    labels = [
        "Zero output",
        "Baseline",
        "Lower threshold + longer filter",
        "Equal trial weight",
        "Early/far training",
        "Stable-target head",
    ]
    values = [means.get(("T2", m), np.nan) for m in methods]
    colors = [
        COLORS[k] for k in ("zero", "baseline", "representation", "weighting", "phase", "phase")
    ]
    ax.barh(np.arange(len(methods)), values, color=colors, height=0.65)
    ax.set_yticks(np.arange(len(methods)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#626C77", lw=0.8)
    ax.axvline(0.05, color="#626C77", lw=0.8, ls="--")
    ax.text(0.055, 4.8, "Recovery\nfloor", fontsize=9, color="#626C77")
    ax.set_xlabel("Mean session R² · unchanged full deployment target")
    ax.set_title("A  T2: fixed comparisons at the same label budget", loc="left", fontweight="bold")
    for i, v in enumerate(values):
        if np.isfinite(v):
            ax.annotate(
                f"{v:.3f}",
                (v, i),
                xytext=(4 if v >= 0 else -4, 0),
                textcoords="offset points",
                ha="right" if v < 0 else "left",
                va="center",
                fontsize=10,
            )
    ax.margins(x=0.25)
    ax = fig.add_subplot(grid[0, 1])
    t2 = [r for r in rows if r["participant"] == "T2"]
    pairs = [
        ("baseline_clean_uniform", "instantaneous"),
        ("baseline_clean_uniform", "onset"),
        ("primary_onset", "instantaneous"),
        ("primary_onset", "onset"),
    ]
    vals = []
    for method, target in pairs:
        vv = [
            r["secondary"][target]["r2"]
            for r in t2
            if r["method"] == method and r["secondary"][target]["r2"] is not None
        ]
        vals.append(np.mean(vv) if vv else np.nan)
    ax.bar(
        range(4),
        vals,
        color=[COLORS["baseline"], COLORS["baseline"], COLORS["phase"], COLORS["phase"]],
        width=0.65,
    )
    ax.set_xticks(
        range(4),
        [
            "Instant target\nInstant head",
            "Stable target\nInstant head",
            "Instant target\nStable head",
            "Stable target\nStable head",
        ],
    )
    ax.axhline(0, color="#626C77", lw=0.8)
    ax.set_ylabel("Mean T2 R² · same early/far bins")
    ax.set_title("B  Does the target definition explain failure?", loc="left", fontweight="bold")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.annotate(
                f"{v:.3f}",
                (i, v),
                xytext=(0, 4 if v >= 0 else -12),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
    ax.margins(y=0.25)
    ax = fig.add_subplot(grid[1, :])
    x = np.arange(len(people))
    width = 0.36
    ax.bar(
        x - width / 2,
        [means[p, "baseline"] for p in people],
        width,
        label="Baseline",
        color=COLORS["baseline"],
    )
    ax.bar(
        x + width / 2,
        [means[p, "selected_all_uniform"] for p in people],
        width,
        label="Source-selected representation",
        color=COLORS["representation"],
    )
    for i, p in enumerate(people):
        for j, m in enumerate(("baseline", "selected_all_uniform")):
            v = [
                r["metrics"]["r2"]
                for r in rows
                if r["participant"] == p and r["method"] == m and r["metrics"]["r2"] is not None
            ]
            offsets = np.linspace(-0.06, 0.06, len(v))
            ax.scatter(
                i + (-width / 2 if j == 0 else width / 2) + offsets,
                v,
                s=22,
                color="#1E2B38",
                zorder=3,
            )
    ax.axhline(0, color="#626C77", lw=0.8)
    ax.set_xticks(x, people)
    ax.set_ylabel("Mean session R²")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.13)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(
        "C  Portability across known participants · dots show every scorable session",
        loc="left",
        fontweight="bold",
    )
    fig.suptitle(
        "T2 failure diagnosis on fresh sessions", fontsize=19, fontweight="bold", x=0.02, ha="left"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(output.with_suffix("." + ext), dpi=180)
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    write_json(
        output.with_suffix(".json"),
        {
            "source_sha256": digest(study_path),
            "code_sha256": digest(__file__),
            "source_role": (
                "Fresh fixed-rank sessions, known participants; no post-test model selection"
            ),
            "figure_files": {ext: digest(output.with_suffix("." + ext)) for ext in ("png", "svg")},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.study, args.output)
