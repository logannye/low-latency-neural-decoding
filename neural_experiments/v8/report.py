"""Generate the complete fresh-result tables from the audited public V8 study."""

# Markdown table rows and paragraph templates are emitted on single lines.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_experiments.v3.study import digest

LABELS = {
    "baseline": "Baseline",
    "baseline_all_equal_trial": "Baseline + equal-trial weights",
    "baseline_clean_uniform": "Baseline + early/far training",
    "baseline_clean_equal_trial": "Baseline + early/far + equal-trial",
    "selected_all_uniform": "TX3/500 ms representation",
    "selected_all_equal_trial": "TX3/500 ms + equal-trial",
    "selected_clean_uniform": "TX3/500 ms + early/far",
    "selected_clean_equal_trial": "TX3/500 ms + early/far + equal-trial",
    "zero": "Zero output",
    "constant": "Purchased-label constant",
    "trialshuffle": "Fixed trial-label rotation",
    "primary_onset": "Stable-reference target head",
}


def fmt(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def report(study_path, output):
    study_path = Path(study_path)
    study = json.loads(study_path.read_text())
    if (
        study.get("study") != "V8"
        or study.get("phase") != "assessment"
        or not study.get("audit_sha256")
    ):
        raise ValueError("audited public V8 assessment required")
    summary = study["summary"]
    methods = summary["methods"]
    baseline = methods["baseline"]
    primary = summary["primary_method"]
    passed = summary["gates"]["useful_T2_primary_recovery"]
    t2 = {
        m: next(p for p in x["participants"] if p["participant"] == "T2")
        for m, x in methods.items()
    }
    lines = [
        "# V8: T2 failure diagnosis on fresh sessions",
        "",
        f"**Predeclared T2 recovery gate: {'PASS' if passed else 'FAIL'}.** The source-selected primary was `{primary}`. "
        "The source screen retained the existing baseline among the neural recipes; no fresh result changed that choice.",
        "",
        f"The fixed cohort contains {summary['selected_sessions']} selected sessions, with {baseline['scored_sessions']} scorable "
        f"sessions across {baseline['n_participants']} known participants. T2's primary mean R² is {fmt(t2[primary]['mean_r2'])}. "
        "Every selected case, including unavailable scores and failed interventions, is retained below and in the public artifacts.",
        "",
        "## What was tested",
        "",
        "The factorial experiment separates two neural representations, all-phase versus early/far training labels, "
        "and uniform versus equal-trial weights. The original continuous cursor-to-target direction remains the primary endpoint. "
        "A dedicated stable-reference target head and its matched instantaneous-target head are also evaluated on exactly the same "
        "early/far bins. Changing that secondary endpoint cannot satisfy the primary gate.",
        "",
        "Every method purchases the same recorded-time interval, seconds 30–150, after 30 seconds of normalization. "
        "Scoring starts at 152 seconds, ends by 600 seconds, and excludes all trials overlapping calibration. "
        "Each session trains its own readout. The recipes were selected on already exposed T2 data, so this is "
        "known-participant session generalization, not zero-shot transfer to a new person.",
        "",
        "## All fixed methods",
        "",
        "| Method | T2 mean R² | Cohort mean R² [participant 95% interval] | Other seven mean R² |",
        "|---|---:|---:|---:|",
    ]
    for method in LABELS:
        x = methods[method]
        ci = x["participant_bootstrap_95"]
        lines.append(
            f"| {LABELS[method]} | {fmt(t2[method]['mean_r2'])} | {fmt(x['mean_r2'])} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(x['non_T2_mean_r2'])} |"
        )
    lines += [
        "",
        "Average sessions within each person, then average people equally. Confidence intervals resample people; "
        "the three-session T2 intervals are descriptive and provide limited uncertainty information. "
        "Other than the locked primary, these are prespecified diagnostic comparisons, not alternate winners selected after testing.",
        "",
        "![Fresh T2 mechanism comparisons and every participant](../figures/v8-diagnosis.png)",
        "",
        "## Every T2 session",
        "",
        "| Session | Baseline | TX3/500 ms | Equal-trial | Early/far | Stable target head | Zero |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ids = sorted(
        {r["session_id"] for r in study["rows"] if r["participant"] == "T2"},
        key=lambda s: int(s.split("_day_")[1].split("_")[0]),
    )
    lookup = {(r["session_id"], r["method"]): r for r in study["rows"]}
    for sid in ids:
        names = (
            "baseline",
            "selected_all_uniform",
            "baseline_all_equal_trial",
            "baseline_clean_uniform",
            "primary_onset",
            "zero",
        )
        lines.append(
            "| "
            + sid
            + " | "
            + " | ".join(fmt(lookup[sid, m]["metrics"]["r2"]) for m in names)
            + " |"
        )
    lines += [
        "",
        "## Factorial attribution",
        "",
        "| Paired effect | T2 mean change in R² | Cohort mean change [participant 95% interval] |",
        "|---|---:|---:|",
    ]
    for name, effect in summary["factorial_effects"].items():
        ci = effect["paired_participant_95"]
        lines.append(
            f"| {name.replace('_', ' ')} | {fmt(effect['T2_mean_effect'], 4)} | {fmt(effect['mean_effect'], 4)} [{fmt(ci[0], 4)}, {fmt(ci[1], 4)}] |"
        )
    lines += [
        "",
        "Main effects average the paired change across the other two factors. Interactions are differences of differences. "
        "These are descriptive mechanism comparisons without multiplicity-adjusted significance claims. "
        "Equal-trial weighting changes duration dominance; it does not collect new trials or balance directions. "
        "Early/far training reduces the amount of usable data within the same purchased interval, so its effect combines phase choice and information quantity.",
        "",
        "## Known-participant portability",
        "",
        "| Person | Selected/scorable | Baseline R² | TX3/500 ms R² |",
        "|---|---:|---:|---:|",
    ]
    for person in baseline["participants"]:
        other = next(
            p
            for p in methods["selected_all_uniform"]["participants"]
            if p["participant"] == person["participant"]
        )
        lines.append(
            f"| {person['participant']} | {person['selected_sessions']}/{person['scored_sessions']} | {fmt(person['mean_r2'])} | {fmt(other['mean_r2'])} |"
        )
    lines += [
        "",
        "## Evidence and interpretation",
        "",
        "The [mechanism diagnosis](DIAGNOSIS.md) documents the source decline, fresh confirmation, trial coverage, target geometry, "
        "feature sparsity and error decomposition. Its post-development scalar-gain oracle is an explicitly nondeployable diagnostic. "
        "It neither changes the selected model nor bounds information in the original recordings.",
        "",
        "The data do not identify actual intention, assistance level, task strategy, electrode biology or a unique causal explanation for failure. "
        "Failure of these finite fixes cannot prove that T2 is undecodable. Conversely, more threshold crossings, positive correlation, "
        "or success on an easier subset cannot establish useful continuous control.",
        "",
        "See the high-level repository README for the fresh mechanism interpretation.",
        "",
        "## Reproduction and provenance",
        "",
        "- [Frozen protocol](PROTOCOL.md) and [source selection](source-selection.json).",
        "- [Data contract](DATA.md), [fresh cohort](../results/v8-cohort.json), and [fresh raw-context audit](../results/v8-fresh-data-audit.json).",
        "- [All fresh aggregate results](../results/v8-study.json) and [independent numeric audit](../results/v8-audit.json).",
        "- [Source results](../results/v8-development-study.json), [source audit](../results/v8-development-audit.json), and [runtime evidence](RUNTIME.md).",
        "- [Reproduction commands](README.md). Raw samples, private traces and fitted coefficients remain outside Git.",
        "",
        f"Public assessment SHA256: `{digest(study_path)}`.",
        f"Private assessment manifest SHA256: `{study['private_manifest_sha256']}`.",
        f"Independent audit SHA256: `{study['audit_sha256']}`.",
        "",
    ]
    Path(output).write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.study, args.output)
