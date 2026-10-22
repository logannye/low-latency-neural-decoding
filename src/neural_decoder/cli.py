"""Small explicit entry points for local H1 development and feature-only replay."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline", help="run fixed S0-S4 forward-day grid")
    baseline.add_argument("--index", type=Path, required=True)
    baseline.add_argument(
        "--config", type=Path, default=Path("neural_experiments/configs/ridge-v1.json")
    )
    baseline.add_argument("--output", type=Path, required=True)
    temporal = commands.add_parser("temporal", help="run the fixed TCN development recipe")
    temporal.add_argument("--index", type=Path, required=True)
    temporal.add_argument(
        "--config", type=Path, default=Path("neural_experiments/configs/tcn-v1.json")
    )
    temporal.add_argument("--output", type=Path, required=True)
    temporal.add_argument("--pilot", action="store_true")
    replay = commands.add_parser("replay", help="replay one feature-only file with a linear model")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--kind", choices=["linear", "tcn"], default="linear")
    replay.add_argument("--features", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "baseline":
        from .train import run_baselines

        run_baselines(
            args.index, args.config, args.output, root=Path(__file__).resolve().parents[2]
        )
    elif args.command == "temporal":
        from .train_temporal import run_temporal

        run_temporal(
            args.index,
            args.config,
            args.output,
            root=Path(__file__).resolve().parents[2],
            pilot=args.pilot,
        )
    else:
        from .replay import replay_features

        replay_features(args.model, args.features, args.output, kind=args.kind)


if __name__ == "__main__":
    main()
