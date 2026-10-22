"""V3 figures from public aggregate metrics only, with explicit case pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "neural-v3-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2464A1"
ORANGE = "#B96528"
TEAL = "#247E78"
PURPLE = "#8061A6"
GRAY = "#697785"
INK = "#213547"
METHODS = (
    "zero",
    "prefix_mean",
    "v2_ridge_daily",
    "tx_daily",
    "tx_sbp_daily",
    "selected_daily",
    "stale_mapping",
)
LABELS = {
    "zero": "Zero vector",
    "prefix_mean": "Prefix mean",
    "v2_ridge_daily": "V2 ridge, daily fit",
    "tx_daily": "Counts, daily fit",
    "tx_sbp_daily": "Counts + power, daily fit",
    "selected_daily": "Source-selected, daily fit",
    "stale_mapping": "Stale mapping (0 fresh labels)",
}
COLORS = {
    "zero": GRAY,
    "prefix_mean": "#8E9CA8",
    "v2_ridge_daily": ORANGE,
    "tx_daily": BLUE,
    "tx_sbp_daily": PURPLE,
    "selected_daily": TEAL,
    "stale_mapping": GRAY,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def person_order(person):
    return (int(person[1:]) if person.startswith("T") and person[1:].isdigit() else 999, person)


def cases(data, method, budget, partition):
    """Extract one finite metric per session, retaining its exact scorer window."""
    output = {}
    for identity in data["models"]:
        if identity["method"] != method or identity["budget_seconds"] != budget:
            continue
        traces = [t for t in identity["traces"] if t["partition"] == partition]
        if len(traces) != 1:
            raise ValueError("each model must retain exactly one requested partition")
        trace = traces[0]
        if trace["metrics"]["r2"] is None:
            continue
        value = float(trace["metrics"]["r2"])
        if not np.isfinite(value):
            raise ValueError("nonfinite aggregate score")
        expected_start = 15100 if partition == "common_suffix" else (budget + 2) * 50
        if trace["start_bin"] != expected_start or not expected_start < trace["stop_bin"] <= 30000:
            raise ValueError("unexpected score window")
        key = (identity["target_participant"], identity["session_id"])
        if key in output:
            raise ValueError("duplicate participant/session score")
        output[key] = {
            "r2": value,
            "signature": (trace["start_bin"], trace["stop_bin"], trace["mask_sha256"]),
        }
    if not output:
        raise ValueError(f"no eligible {method}/{budget}/{partition} cases")
    return output


def person_means(rows, selected=None):
    people = defaultdict(list)
    for key in rows if selected is None else selected:
        people[key[0]].append(rows[key]["r2"])
    return {
        person: {"r2": float(np.mean(values)), "n_sessions": len(values)}
        for person, values in sorted(people.items(), key=lambda pair: person_order(pair[0]))
    }


def checked_group(data, method, budget=120, partition="primary"):
    rows = cases(data, method, budget, partition)
    means = person_means(rows)
    groups = [
        g
        for g in data["summary"]["groups"]
        if g["method"] == method and g["budget_seconds"] == budget and g["partition"] == partition
    ]
    if len(groups) != 1:
        raise ValueError("missing or duplicate published group")
    group = groups[0]
    published = {p["participant"]: p for p in group["people"]}
    if set(published) != set(means):
        raise ValueError("group participant inventory differs from case metrics")
    for person, values in means.items():
        if published[person]["n_sessions"] != values["n_sessions"] or not np.isclose(
            published[person]["r2"], values["r2"], rtol=0, atol=1e-10
        ):
            raise ValueError("group mean/count differs from case metrics")
    if not np.isclose(
        group["mean_r2"], np.mean([p["r2"] for p in means.values()]), rtol=0, atol=1e-10
    ):
        raise ValueError("group does not give people equal weight")
    return rows, means, group


def match_signatures(lookups, selected):
    for key in selected:
        if len({lookup[key]["signature"] for lookup in lookups}) != 1:
            raise ValueError("paired scores use different masks or windows")


def primary_comparison(data):
    selected, selected_people, _ = checked_group(data, "selected_daily")
    reference, reference_people, _ = checked_group(data, "v2_ridge_daily")
    if set(selected) != set(reference):
        raise ValueError("primary matched comparison has different eligible sessions")
    match_signatures([selected, reference], set(selected))
    stale, stale_people, _ = checked_group(data, "stale_mapping")
    common = set(stale) & set(selected)
    match_signatures([selected, stale], common)
    return {
        "selected": selected_people,
        "reference": reference_people,
        "stale": stale_people,
        "n_people": len(selected_people),
        "n_sessions": len(selected),
        "stale_n_people": len(stale_people),
        "stale_n_sessions": len(stale),
    }


def paired_budgets(data):
    methods = ("tx_daily", "tx_sbp_daily", "selected_daily")
    lookups = {
        (method, budget): cases(data, method, budget, "common_suffix")
        for method in methods
        for budget in (120, 300)
    }
    shared = set.intersection(*(set(v) for v in lookups.values()))
    union = set.union(*(set(v) for v in lookups.values()))
    if not shared:
        raise ValueError("no common participant/session set for budget comparison")
    match_signatures(list(lookups.values()), shared)
    rows = []
    for (method, budget), lookup in lookups.items():
        means = person_means(lookup, shared)
        rows.append(
            {
                "method": method,
                "budget_seconds": budget,
                "mean_r2": float(np.mean([p["r2"] for p in means.values()])),
                "people": means,
            }
        )
    return {
        "partition": "common_suffix",
        "window_seconds": [302, 600],
        "n_people": len({key[0] for key in shared}),
        "n_sessions": len(shared),
        "excluded_unpaired_sessions": len(union - shared),
        "sessions": [{"participant": p, "session_id": s} for p, s in sorted(shared)],
        "rows": rows,
    }


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#B0BAC4",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "axes.titleweight": "bold",
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "neural-v3-aggregate-figures",
            "legend.frameon": False,
        }
    )


def clean(ax, *, horizontal=True):
    ax.grid(axis="x" if horizontal else "y", color="#E0E5EA", lw=0.7)
    ax.tick_params(length=0, pad=6)
    ax.spines["left"].set_visible(False)
    if horizontal:
        ax.axvline(0, color=GRAY, lw=1, zorder=1)
    else:
        ax.axhline(0, color=GRAY, lw=1, zorder=1)


def footer(fig, lines, source):
    fig.text(0.065, 0.105, lines, fontsize=9.1, color=GRAY, va="bottom", linespacing=1.5)
    fig.text(0.065, 0.025, source, fontsize=8.1, color=GRAY)


def save(fig, output, name):
    paths = []
    for extension in ("png", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=180, metadata={"Date": None} if extension == "svg" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def render(data, output, *, source="Public aggregate evidence"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    style()
    primary = primary_comparison(data)
    budgets = paired_budgets(data)
    groups = {method: checked_group(data, method) for method in METHODS}
    paths = []
    people = sorted(set(primary["selected"]) | set(primary["stale"]), key=person_order)
    y = np.arange(len(people))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.4), gridspec_kw={"width_ratios": [1.65, 1]})
    fig.subplots_adjust(left=0.105, right=0.965, top=0.76, bottom=0.27, wspace=0.32)
    fig.suptitle(
        "Fresh-session accuracy, with calibration made explicit",
        x=0.065,
        y=0.965,
        ha="left",
        fontsize=20,
        weight="bold",
    )
    fig.text(
        0.065,
        0.905,
        "Previously studied people; new session evidence. "
        "Absolute R², equal sessions within each person.",
        color=GRAY,
    )
    for i, person in enumerate(people):
        if person in primary["selected"]:
            a, b = primary["reference"][person]["r2"], primary["selected"][person]["r2"]
            axes[0].plot([a, b], [i, i], color="#BBC5CD", lw=2, zorder=1)
            axes[0].scatter(a, i, color=ORANGE, marker="s", s=45, zorder=3)
            axes[0].scatter(b, i, color=TEAL, marker="o", s=48, zorder=3)
        if person in primary["stale"]:
            axes[1].scatter(primary["stale"][person]["r2"], i, color=PURPLE, marker="D", s=43)
    axes[0].scatter([], [], color=TEAL, label="Source-selected daily fit", s=48)
    axes[0].scatter([], [], color=ORANGE, marker="s", label="V2 ridge daily fit", s=45)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=1, fontsize=9)
    axes[0].set_title("Matched: 120 s of fresh-session calibration", pad=50, fontsize=12)
    axes[1].set_title("Stale mapping: no fresh labels", pad=50, fontsize=12)
    axes[1].text(
        0,
        1.025,
        "Separate x scale; different label budget",
        transform=axes[1].transAxes,
        fontsize=9,
        color=GRAY,
    )
    for ax in axes:
        ax.set_yticks(y, people)
        ax.set_ylim(len(people) - 0.5, -0.7)
        ax.set_xlabel("Variance-weighted R²")
        clean(ax)
        ax.margins(x=0.15)
    gate = "passed" if data["summary"]["operational_gate"]["passed"] else "not passed"
    footer(
        fig,
        f"Daily comparison: {primary['n_people']} people, {primary['n_sessions']} scored sessions; "
        "[122, 600) s, truncated at recording end.\n"
        f"Stale mapping: {primary['stale_n_people']} people, "
        f"{primary['stale_n_sessions']} sessions; "
        "earliest development-session calibration only.\n"
        f"Offline engineering gate: {gate}. This is not a clinical-utility threshold; "
        "zero R² is the scoring reference.",
        source,
    )
    paths += save(fig, output, "v3-participant-accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.22, right=0.96, top=0.77, bottom=0.36, wspace=0.42)
    fig.suptitle(
        "Absolute performance and a matched calibration-budget comparison",
        x=0.065,
        y=0.965,
        ha="left",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.065,
        0.91,
        "Equal session means within each person, then equal people. "
        "Budget curves use only the same [302, 600) s suffix.",
        color=GRAY,
    )
    for i, method in enumerate(METHODS):
        rows, means, group = groups[method]
        low, high = group["r2_bootstrap_95"]
        center = group["mean_r2"]
        axes[0].plot([low, high], [i, i], color=COLORS[method], marker="|", markersize=6)
        axes[0].scatter(center, i, color=COLORS[method], s=36, zorder=3)
    axes[0].axhspan(len(METHODS) - 1.45, len(METHODS) - 0.55, color="#F0F2F4", zorder=0)
    axes[0].set_yticks(
        range(len(METHODS)),
        [
            f"{LABELS[m]}\n{len(groups[m][1])} people · {len(groups[m][0])} sessions"
            for m in METHODS
        ],
        fontsize=9.2,
    )
    axes[0].set_ylim(len(METHODS) - 0.4, -0.8)
    axes[0].set_title("120 s calibration; primary window", pad=18, fontsize=12)
    axes[0].set_xlabel("R², with descriptive participant-bootstrap 95% interval")
    clean(axes[0])
    for method, marker in (("tx_daily", "s"), ("tx_sbp_daily", "^"), ("selected_daily", "o")):
        values = [
            next(
                r["mean_r2"]
                for r in budgets["rows"]
                if r["method"] == method and r["budget_seconds"] == b
            )
            for b in (120, 300)
        ]
        axes[1].plot(
            [120, 300],
            values,
            color=COLORS[method],
            marker=marker,
            markersize=4.5 if method == "selected_daily" else 7.5,
            markerfacecolor=COLORS[method] if method == "selected_daily" else "none",
            linestyle="--" if method == "tx_sbp_daily" else "-",
            lw=2,
            label=LABELS[method],
        )
    axes[1].set_xticks([120, 300], ["120 s", "300 s"])
    axes[1].set_xlim(85, 335)
    axes[1].set_xlabel("Fresh-session calibration prefix")
    axes[1].set_ylabel("R² on the identical common suffix")
    axes[1].set_title("Matched sessions, people, masks and window", pad=18, fontsize=12)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8.8)
    clean(axes[1], horizontal=False)
    footer(
        fig,
        "Left: [122, 600) s at 120 s calibration. "
        "Shaded stale mapping uses no fresh-session labels.\n"
        f"Right: {budgets['n_people']} people and {budgets['n_sessions']} identical sessions "
        "across both budgets and all three curves; "
        f"{budgets['excluded_unpaired_sessions']} unpaired sessions excluded.\n"
        "Primary 120 s and primary 300 s windows are never compared as a label-budget effect. "
        "Intervals are descriptive, not clinical claims.",
        source,
    )
    paths += save(fig, output, "v3-methods-and-budget")
    return {
        "primary": primary,
        "budget_comparison": budgets,
        "figures": [{"name": path.name, "sha256": digest(path)} for path in paths],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    result = render(
        data,
        args.output,
        source=f"Source: {args.results.name} · SHA256 {digest(args.results)[:12]}",
    )
    result.update(source_sha256=digest(args.results), figure_source_sha256=digest(__file__))
    (args.output / "v3-figures.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result["figures"], indent=2))


if __name__ == "__main__":
    main()
