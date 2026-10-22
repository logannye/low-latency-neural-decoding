"""Separate new-session and T3-continuation figures from audited aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from neural_experiments.v3.study import digest, write_json  # noqa: E402


def make(study_path, output):
    result = json.loads(Path(study_path).read_text())
    if result["independent_audit"]["status"] != "PASS":
        raise ValueError("audited aggregate required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "v4-locked-study",
        }
    )
    files = []
    definitions = [
        (
            "new_sessions",
            "v4-reduced-calibration",
            "Fresh sessions across eight known participants",
            ["T2", "T5", "T6", "T7", "T8", "T9", "T10", "T11"],
            [
                ("selected_prior0", "History + 0 s labels", "#8ca5a2"),
                ("selected_prior30", "History + 30 s labels", "#167d8d"),
                ("v3_daily120", "Daily 120 s labels", "#d9793b"),
            ],
            "Equal-session means per person; all methods score the same suffix from 122 s.",
        ),
        (
            "t3_blocks",
            "v4-t3-continuation",
            "T3: untouched later acquisition blocks",
            [f"T3_day_{d}_decoding" for d in (36, 37, 43, 44, 62, 79, 100, 406)],
            [
                ("zero", "Zero output", "#9ba1a6"),
                ("targeted_daily120", "Targeted 120 s labels", "#167d8d"),
                ("v3_daily120", "V3 daily 120 s labels", "#d9793b"),
            ],
            "Known sessions, fresh blocks; eight session scores within one person.",
        ),
    ]
    for domain, stem, title, units, arms, caption in definitions:
        methods = result["summary"]["domains"][domain]["methods"]
        fig, ax = plt.subplots(figsize=(11.5, 5.6))
        positions = np.arange(len(units))
        for offset, (method, label, color) in enumerate(arms):
            values = {r["unit"]: r["r2"] for r in methods[method]["units"]}
            ax.bar(
                positions + (offset - 1) * 0.24,
                [values.get(u, np.nan) for u in units],
                width=0.22,
                label=label,
                color=color,
            )
        ax.axhline(0, color="#555b61", linewidth=0.8)
        ax.set_xticks(
            positions,
            [u.split("_day_")[1].split("_")[0] if domain == "t3_blocks" else u for u in units],
        )
        ax.set_xlabel("Post-implant day" if domain == "t3_blocks" else "Participant")
        ax.set_ylabel("Variance-weighted R²")
        ax.set_title(title, loc="left", fontweight="bold", pad=18)
        ax.legend(frameon=False, loc="best", fontsize=9)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
        fig.text(0.09, 0.025, caption, fontsize=9, color="#555b61")
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        for extension in ("png", "svg"):
            path = output / f"{stem}.{extension}"
            fig.savefig(path, dpi=180, metadata={"Date": None} if extension == "svg" else None)
            if extension == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
                )
            files.append({"path": path.name, "sha256": digest(path)})
        plt.close(fig)
    write_json(output / "v4-figures.json", {"study_sha256": digest(study_path), "files": files})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    make(a.study, a.output)


if __name__ == "__main__":
    main()
