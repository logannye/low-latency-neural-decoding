"""Publication figures from aggregate results only; no recording arrays are loaded."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results"
FIG = ROOT / "figures"


def read(name):
    return json.loads((DATA / name).read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
    svg = FIG / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "neural-decoding-v1",
        }
    )
    later = read("final-later.json")
    models = {entry["model"]["name"]: entry["scores"] for entry in later["models"]}
    days = [f"S{i}" for i in range(6, 13)]
    ys = np.array(
        [
            [models[f"tcn-seed-{s}"]["per_day"][d]["r2_variance_weighted"] for d in days]
            for s in range(3)
        ]
    )
    ridge = np.array([models["ridge"]["per_day"][d]["r2_variance_weighted"] for d in days])
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for i, values in enumerate(ys):
        ax.plot(
            days,
            values,
            color="#6c8dca",
            alpha=0.45,
            lw=1,
            marker=".",
            label="TCN individual seeds" if i == 0 else None,
        )
    ax.plot(days, ys.mean(0), "-o", color="#214f96", lw=2.2, label="TCN mean of 3 seeds")
    ax.plot(days, ridge, "-s", color="#b5682e", lw=2, label="Causal ridge")
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set(
        ylabel="Day variance-weighted R²",
        xlabel="Held-out recording day",
        title="Frozen recipe on seven later H1 days",
    )
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.text(
        0,
        -0.23,
        "Local public-calibration assessment · one participant · no new-day labels",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    save(fig, "held-out-quality")
    runtime = read("runtime.json")["runs"]
    names = ["torch_window", "numpy_incremental", "onnx_incremental"]
    labels = ["PyTorch window", "NumPy incremental", "ONNX incremental"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, mode in zip(axes, ["fast", "paced"], strict=True):
        for i, name in enumerate(names):
            values = [
                r["step_service"]["p99_ms"]
                for r in runtime
                if r["mode"] == mode and r["implementation"] == name
            ]
            ax.scatter(
                np.full(len(values), i) + np.linspace(-0.08, 0.08, len(values)),
                values,
                color=["#858585", "#214f96", "#3b8874"][i],
                s=50,
                zorder=3,
            )
            ax.hlines(np.median(values), i - 0.2, i + 0.2, color="black", lw=2)
        ax.axhline(2, color="#b5682e", ls="--", lw=1, label="2 ms project goal")
        ax.set_xticks(range(3), labels, rotation=20, ha="right")
        ax.set(
            title="Fast sequential replay" if mode == "fast" else "20 ms paced replay",
            yscale="log",
            ylabel="Complete-step p99 (ms)",
        )
        ax.grid(axis="y", alpha=0.15)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Same float32 weights · batch one · one CPU thread · M4 Max", fontsize=12)
    save(fig, "runtime")
    calibration = read("calibration.json")["results"]
    budgets = [0, 5, 15, 30]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for model, color, label in [
        ("ridge", "#b5682e", "Ridge"),
        ("tcn", "#214f96", "TCN, 3-seed mean"),
    ]:
        values = []
        for budget in budgets:
            scores = [
                r["scores"]["r2_mean"]
                for r in calibration
                if r["prefix_seconds_per_file"] == budget
                and (r["model"] == "ridge" if model == "ridge" else r["model"].startswith("tcn-"))
            ]
            values.append(scores)
        mean = [np.mean(x) for x in values]
        ax.plot(budgets, mean, "-o", color=color, label=label, lw=2)
        if model == "tcn":
            ax.fill_between(
                budgets,
                [min(x) for x in values],
                [max(x) for x in values],
                color=color,
                alpha=0.12,
                label="TCN seed range",
            )
    ax.set(
        xlabel="Nominal labeled prefix seconds per file",
        ylabel="Mean day R² on common suffix",
        title="Separate supervised intercept calibration",
        xticks=budgets,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0,
        -0.25,
        "Every arm scores after 30 s · two files/day → 0, 10, 30, 60 labeled s/day",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    save(fig, "calibration")


if __name__ == "__main__":
    main()
