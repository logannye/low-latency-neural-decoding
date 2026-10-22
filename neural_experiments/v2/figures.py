"""Scientific V2 figures from public aggregate JSON only; no sample/model arrays.

Example:
  python -m neural_experiments.v2.figures --accuracy neural_experiments/results/v2-accuracy.json
Optional --transfer and --runtime add their figures when completed receipts exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "neural-v2-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BLUE = "#2464A1"
ORANGE = "#B96528"
TEAL = "#247E78"
PURPLE = "#8061A6"
INK = "#213547"
GRAY = "#697785"
METHODS = ("ridge", "scratch", "pretrained_frozen_core", "pretrained_finetune")
METHOD_LABELS = (
    "Causal ridge",
    "Neural model from scratch",
    "Pretrained core, frozen",
    "Pretrained core, fine-tuned",
)
METHOD_COLORS = (GRAY, ORANGE, BLUE, TEAL)
PEOPLE = ("T2", "T3", "T5", "T6", "T7", "T8", "T9", "T10", "T11")


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected an aggregate JSON object")
    return value


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#A6B0BA",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "neural-decoding-v2-public-aggregates",
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )


def source_label(path):
    path = Path(path).resolve()
    name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"Source: {name}  |  SHA256 {digest[:12]}"


def frame(fig, title, kicker, footer, source):
    fig.text(0.075, 0.965, kicker, fontsize=10, color=GRAY, weight="bold", ha="left")
    fig.text(0.075, 0.905, title, fontsize=22, color=INK, weight="bold", ha="left")
    fig.text(0.075, 0.075, footer, fontsize=9.2, color=GRAY, va="bottom", linespacing=1.5)
    fig.text(0.075, 0.025, source_label(source), fontsize=8.2, color=GRAY, va="bottom")


def clean_axis(ax):
    ax.grid(axis="y", color="#DCE2E7", lw=0.7)
    ax.tick_params(length=0, pad=7)
    ax.spines["left"].set_visible(False)


def save(fig, output, stem):
    paths = []
    for extension in ("png", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=180, metadata={"Date": None} if extension == "svg" else None)
        paths.append(str(path))
    plt.close(fig)
    return paths


def h1_mean(rows):
    """Equal day means after seed averaging; require the complete 3x3 grid."""
    keys = [(r["day"], r["seed"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != {
        (d, s) for d in ("S2", "S3", "S4") for s in range(3)
    }:
        raise ValueError("H1 figure requires all three seeds on all three development days")
    days = {
        day: float(np.mean([r["scores"]["r2_mean"] for r in rows if r["day"] == day]))
        for day in ("S2", "S3", "S4")
    }
    return float(np.mean(list(days.values()))), days


def accuracy_figures(path, output):
    data = read(path)
    if (
        data["execution"]["status"] != "complete"
        or data["independent_trace_audit"]["status"] != "pass"
    ):
        raise ValueError("H1 figures require complete, independently audited aggregate evidence")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    fig.subplots_adjust(left=0.075, right=0.955, top=0.765, bottom=0.245, wspace=0.28)
    fractions = [0.25, 0.5, 1.0]
    for condition, label, color, marker in (
        ("same_day", "Same-day training", BLUE, "o"),
        ("earlier_days", "Earlier-day training", ORANGE, "s"),
    ):
        means = [
            h1_mean(
                [
                    r
                    for r in data["diagnostics"]
                    if r["condition"] == condition and r["fraction"] == fraction
                ]
            )[0]
            for fraction in fractions
        ]
        axes[0].plot(
            np.array(fractions) * 100, means, color=color, marker=marker, lw=2.6, ms=7, label=label
        )
        axes[0].annotate(
            f"{means[-1]:.3f}",
            (100, means[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=11,
            va="center",
            weight="bold",
        )
    axes[0].set(
        title="A   Matched exposure, later recording scored",
        xlabel="Matched training exposure (% of first recording)",
        ylabel="Mean day R²",
        xticks=[25, 50, 100],
        xlim=(19, 117),
    )
    axes[0].axhline(0, color=GRAY, lw=0.8, ls=(0, (3, 3)))
    axes[0].margins(y=0.22)
    axes[0].legend(loc="upper left", fontsize=10)
    for method, label, color, marker, linestyle in (
        ("unadapted", "No calibration", GRAY, "o", "--"),
        ("intercept", "Intercept correction", ORANGE, "s", "-"),
        ("ridge_delta_head", "Calibrated readout", BLUE, "D", "-"),
    ):
        means = [
            h1_mean(
                [
                    r
                    for r in data["calibration"]
                    if r["arm"] == "rf15_standardized"
                    and r["method"] == method
                    and r["budget_seconds"] == budget
                ]
            )[0]
            for budget in (0, 10, 30)
        ]
        axes[1].plot(
            [0, 10, 30], means, color=color, marker=marker, lw=2.5, ms=6, ls=linestyle, label=label
        )
        axes[1].annotate(
            f"{means[-1]:.3f}",
            (30, means[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=11,
            va="center",
            weight="bold",
        )
    axes[1].set(
        title="B   Same suffix scored for every budget",
        xlabel="Labeled calibration prefix (seconds per recording)",
        ylabel="Mean day R²",
        xticks=[0, 10, 30],
        xlim=(-1.5, 36),
        ylim=(0.105, 0.35),
    )
    axes[1].legend(loc="upper left", fontsize=10)
    for ax in axes:
        clean_axis(ax)
        ax.set_title(ax.get_title(), loc="left", fontsize=12.5, pad=15)
        ax.set_title("")
    frame(
        fig,
        "Recording drift and brief labeled calibration",
        "H1  /  ONE HUMAN  /  DEVELOPMENT ONLY",
        "Variance-weighted R² over all S2–S4 days and all three seeds; days weighted equally. "
        "Repeated observations from one participant.\n"
        "A: duration and labels matched; behavior differences remain.  "
        "B: fixed 15-bin model; common suffix starts at 30 s.",
        path,
    )
    outputs = save(fig, output, "v2-h1-calibration")

    arms = [
        ("rf15_standardized", "15-bin history · standardized loss"),
        ("rf15_source_units", "15-bin history · source-unit loss"),
        ("rf63_standardized", "63-bin history · standardized loss"),
        ("rf63_source_units", "63-bin history · source-unit loss"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.subplots_adjust(left=0.34, right=0.86, top=0.72, bottom=0.28)
    for i, (arm, label) in enumerate(arms):
        mean, day_means = h1_mean([r for r in data["main"] if r["arm"] == arm])
        ax.barh(
            i, mean, height=0.43, color=BLUE if arm == "rf15_standardized" else "#B8C6D4", zorder=2
        )
        for j, (day, marker) in enumerate(zip(("S2", "S3", "S4"), ("o", "s", "^"), strict=True)):
            ax.scatter(
                day_means[day],
                i,
                marker=marker,
                s=36,
                facecolors="white",
                edgecolors=INK,
                lw=1.2,
                zorder=4,
                label=f"{day} day mean" if i == 0 else None,
            )
        ax.text(
            1.14,
            i,
            f"{mean:.3f}",
            fontsize=11,
            va="center",
            ha="right",
            weight="bold",
            transform=ax.get_yaxis_transform(),
        )
    ax.set(
        yticks=range(4),
        yticklabels=[label for _, label in arms],
        xlabel="Day variance-weighted R²",
        xlim=(-0.58, 0.38),
    )
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.grid(axis="x", color="#DCE2E7", lw=0.7)
    ax.tick_params(length=0, pad=7)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.06), ncol=3, fontsize=9)
    ax.text(1.14, 1.07, "MEAN", fontsize=9, ha="right", color=GRAY, transform=ax.transAxes)
    frame(
        fig,
        "All model comparisons, including worse outcomes",
        "H1  /  ONE HUMAN  /  DEVELOPMENT ONLY",
        "Bars: equal-day mean over all three seeds. "
        "Points: every development day after seed averaging.\n"
        "Epoch count fixed on S1. All three alternatives failed "
        "the prespecified gain/harm rule.",
        path,
    )
    outputs += save(fig, output, "v2-h1-arms")
    return outputs


def transfer_curves(data):
    summary = data.get("summary", data)
    if summary["status"] != "complete":
        raise ValueError("Transfer figures require a complete aggregate run")
    budgets = sorted({g["budget_seconds"] for g in summary["groups"]})
    lookup = {}
    for method in METHODS:
        for budget in budgets:
            matches = [
                g
                for g in summary["groups"]
                if g["method"] == method
                and g["budget_seconds"] == budget
                and g["partition"] == "future_day"
            ]
            if len(matches) != 1:
                raise ValueError("Missing or duplicate transfer method/budget group")
            lookup[(method, budget)] = {
                p["participant"]: p["r2"]
                for p in matches[0]["people"]
                if p["r2"] is not None and p["n_seeds"] == (1 if method == "ridge" else 3)
            }
    common = set.intersection(*(set(values) for values in lookup.values()))
    common = [person for person in PEOPLE if person in common]
    if not common:
        raise ValueError("No common participant cohort across all methods and calibration budgets")
    means = {
        method: [
            float(np.mean([lookup[(method, budget)][p] for p in common])) for budget in budgets
        ]
        for method in METHODS
    }
    return summary, budgets, common, means


def transfer_figure(path, output):
    summary, budgets, common, means = transfer_curves(read(path))
    contrast = summary["primary_contrast"]
    effects = {r["participant"]: r["difference"] for r in contrast["effects"]}
    people = [p for p in PEOPLE if p in effects]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.9), gridspec_kw={"width_ratios": [1.04, 1]})
    fig.subplots_adjust(left=0.075, right=0.95, top=0.765, bottom=0.255, wspace=0.32)
    for method, label, color, marker in zip(
        METHODS, METHOD_LABELS, METHOD_COLORS, ("s", "^", "o", "D"), strict=True
    ):
        axes[0].plot(budgets, means[method], color=color, marker=marker, lw=2.4, ms=6, label=label)
    axes[0].set(
        title=f"A   Calibration curves · same {len(common)} people",
        xticks=budgets,
        xlabel="Labeled target-person prefix (observed seconds)",
        ylabel="Mean future-day R²",
    )
    axes[0].margins(y=0.25, x=0.08)
    axes[0].axhline(0, lw=0.8, color=GRAY, ls=(0, (3, 3)))
    axes[0].legend(fontsize=9.4, loc="best")
    clean_axis(axes[0])
    values = [effects[p] for p in people]
    for i, value in enumerate(values):
        color = BLUE if value >= 0 else ORANGE
        axes[1].hlines(i, 0, value, color=color, lw=1.8)
        axes[1].plot(value, i, "o", color=color, ms=6)
    if people:
        mean, interval = contrast["mean_difference"], contrast["bootstrap_percentile_95"]
        y = len(people) + 0.65
        axes[1].hlines(y, *interval, color=INK, lw=2.4)
        axes[1].plot(mean, y, "D", color=INK, ms=8)
        axes[1].axhline(len(people) - 0.2, lw=0.7, color="#DCE2E7")
        axes[1].set_yticks([*range(len(people)), y], [*people, "Mean · 95% CI"])
        limits = np.array([*values, *interval, 0.0])
        span = max(float(np.ptp(limits)), 0.04)
        axes[1].set_xlim(float(limits.min()) - span * 0.18, float(limits.max()) + span * 0.23)
        axes[1].text(
            0.02,
            -0.18,
            f"Mean ΔR² {mean:+.3f}  [{interval[0]:+.3f}, {interval[1]:+.3f}]",
            transform=axes[1].transAxes,
            fontsize=10,
            color=INK,
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "No complete paired participant effects",
            transform=axes[1].transAxes,
            ha="center",
        )
    axes[1].axvline(0, color=GRAY, lw=0.9, ls=(0, (3, 3)))
    axes[1].invert_yaxis()
    axes[1].set(
        title=f"B   Registered 120 s contrast · {len(people)} people",
        xlabel="ΔR²: frozen pretrained core − scratch",
    )
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(length=0, pad=7)
    for ax in axes:
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(title, loc="left", fontsize=12.5, pad=15)
    primary_groups = {
        group["method"]: group
        for group in summary["groups"]
        if group["partition"] == "future_day" and group["budget_seconds"] == 120
    }
    frozen, scratch = (
        primary_groups["pretrained_frozen_core"],
        primary_groups["scratch"],
    )
    absolute_note = (
        f"120 s, all {len(people)} people: frozen core R² {frozen['mean_person_r2']:.3f}; "
        f"scratch {scratch['mean_person_r2']:.3f}."
    )
    if all(row["r2"] < 0 for group in (frozen, scratch) for row in group["people"]):
        absolute_note += " Every person's mean R² is negative."
    fig.text(0.075, 0.84, absolute_note, fontsize=11, color=ORANGE)
    frame(
        fig,
        "Transfer to held-out people with limited calibration",
        "BRAINGATE  /  NINE HUMAN PARTICIPANTS  /  BOUNDED OFFLINE STUDY",
        f"Curves: fixed cohort ({', '.join(common)}); days, seeds, then people averaged. "
        f"Paired panel: all {len(people)} eligible people.\n"
        "95% interval resamples people. Cursor-to-target direction endpoint; "
        "supervised offline transfer with target calibration.",
        path,
    )
    return save(fig, output, "v2-participant-transfer")


def runtime_figure(path, output):
    data = read(path)
    runs = data["blocks"]
    period = data["protocol"]["period_ms"]
    if not runs or period <= 0:
        raise ValueError("Runtime receipt requires measured blocks and a positive release period")
    names = list(dict.fromkeys(r["implementation"] for r in runs))
    labels = {"numpy_incremental": "Incremental NumPy", "torch_window": "PyTorch window"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.75, bottom=0.265, wspace=0.25)

    def dots(ax, values, x, color, marker, label):
        if not values:
            return
        if min(values) <= 0 or not np.isfinite(values).all():
            raise ValueError(
                "Log-latency figure requires finite, strictly positive measured latency"
            )
        ax.scatter(
            x + np.linspace(-0.04, 0.04, len(values)),
            values,
            color=color,
            marker=marker,
            s=50,
            label=label,
            zorder=3,
        )
        ax.hlines(float(np.median(values)), x - 0.09, x + 0.09, color=color, lw=2)

    for i, name in enumerate(names):
        for mode, offset, color, marker, label in (
            ("fast", -0.13, BLUE, "o", "Fast replay"),
            ("paced", 0.13, ORANGE, "s", f"{period:g} ms paced replay"),
        ):
            values = [
                r["service_p99_ms"]
                for r in runs
                if r["mode"] == mode and r["implementation"] == name
            ]
            dots(axes[0], values, i + offset, color, marker, label if i == 0 else None)
        for field, offset, color, marker, label in (
            ("service_p99_ms", -0.13, BLUE, "o", "Service time"),
            ("release_to_completion_p99_ms", 0.13, PURPLE, "D", "Release to completion"),
        ):
            values = [
                r[field] for r in runs if r["mode"] == "paced" and r["implementation"] == name
            ]
            dots(axes[1], values, i + offset, color, marker, label if i == 0 else None)
    axes[0].axhline(2, color=GRAY, ls="--", lw=1.1, label="2 ms engineering target")
    axes[1].axhline(period, color=GRAY, ls="--", lw=1.1, label=f"{period:g} ms deadline")
    for ax, title in zip(
        axes, ("A   Complete-step service", "B   Paced deadline accounting"), strict=True
    ):
        ax.set(
            xticks=range(len(names)),
            xticklabels=[labels.get(n, n) for n in names],
            ylabel="p99 latency (ms, log scale)",
            yscale="log",
            xlim=(-0.5, len(names) - 0.5),
        )
        ax.set_title(title, loc="left", pad=15, fontsize=12.5)
        ax.legend(
            fontsize=9,
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=1,
        )
        clean_axis(ax)
    paced = [r for r in runs if r["mode"] == "paced"]
    accounting = "No paced blocks were recorded."
    if paced:
        queues = [r["queue_p99_ms"] for r in paced]
        accounting = (
            f"Paced queue p99: {min(queues):.3f}–{max(queues):.3f} ms; "
            f"deadline misses: {sum(r['deadline_misses'] for r in paced)} / "
            f"{sum(r['n'] for r in paced):,} releases."
        )
    title = (
        "H1 calibrated readout"
        if "h1" in Path(path).stem.lower()
        else "Participant-transfer decoder"
    )
    frame(
        fig,
        f"Feature-ready runtime · {title}",
        "FEATURE-READY CPU REPLAY  /  BATCH ONE",
        "Dots: every measured block; horizontal marks: medians. " + accounting + "\n"
        f"{data['parameter_count']:,} parameters; model SHA256 {data['model_sha256'][:12]}. "
        "Acquisition latency and hard real-time behavior remain unverified.",
        path,
    )
    stem = Path(path).stem if Path(path).stem.startswith("v2-") else "v2-runtime"
    return save(fig, output, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--outputdir", type=Path, default=ROOT / "neural_experiments/figures")
    args = parser.parse_args()
    args.outputdir.mkdir(parents=True, exist_ok=True)
    style()
    outputs = accuracy_figures(args.accuracy, args.outputdir)
    if args.transfer:
        outputs += transfer_figure(args.transfer, args.outputdir)
    if args.runtime:
        outputs += runtime_figure(args.runtime, args.outputdir)
    print(json.dumps({"figures": outputs, "input_scope": "public aggregate JSON only"}, indent=2))


if __name__ == "__main__":
    main()
