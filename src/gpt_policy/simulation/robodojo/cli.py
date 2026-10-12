"""Simulation CLI; offline checks never initialize hardware, Isaac or models."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import check, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--task", default="classify_objects")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--layouts", help="Predetermined comma-separated layout IDs")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--suite", help="Suite JSON path or robodojo_10tasks / classify_five"
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--campaign", action="store_true", help="Freeze even a single case"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    gpus = [int(x) for x in args.gpus.split(",")]
    if args.repetitions < 1:
        raise ValueError("Repetitions must be positive")
    if args.suite:
        path = Path(args.suite)
        if not path.is_file():
            path = (
                Path(__file__).resolve().parents[4]
                / "configs/evaluation"
                / (args.suite + ".json")
            )
        specs = json.loads(path.read_text())["episodes"]
    else:
        layouts = (
            [int(x) for x in args.layouts.split(",")] if args.layouts else [args.layout]
        )
        specs = [
            {"task": args.task, "seed": args.seed, "layout": layout}
            for layout in layouts
            for _ in range(args.repetitions)
        ]
    if args.check:
        reports = [check(config, s["task"], s["seed"], s["layout"]) for s in specs]
        print(
            json.dumps(
                reports[0] if len(reports) == 1 else reports,
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.prepare_only:
        if len(specs) != 1:
            raise ValueError("Prepare one context at a time")
        from ...input import resolve_run_input
        from ...input.demonstration import save_input

        prepared = config.get("prepared_inputs", {}).get(specs[0]["task"])
        if not prepared:
            raise ValueError("No prepared context for selected task")
        if args.prepare_only.exists():
            raise ValueError("Output already exists")
        save_input(
            resolve_run_input(None, Path(prepared), config.get("model", "gpt-6-astra")),
            args.prepare_only,
        )
        return
    queued = (
        args.campaign or args.resume or args.suite or len(specs) > 1 or len(gpus) > 1
    )
    if args.resume and not args.output:
        raise ValueError("Resume needs the original --output directory")
    suffix = "campaign" if queued else ("smoke" if args.smoke else "development")
    output = args.output or Path(config["output_root"]) / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + specs[0]["task"]
        + "_"
        + suffix
    )
    if queued:
        from .campaign import run_campaign

        result = run_campaign(
            config, specs, output, gpus, smoke=args.smoke, resume=args.resume
        )
    else:
        spec = specs[0]
        evidence = check(config, spec["task"], spec["seed"], spec["layout"])
        if args.smoke:
            from .smoke import run_smoke as execute
        else:
            from .runner import run_episode as execute
        result = execute(config, evidence, output, gpus[0])
    print(json.dumps({"output": str(output), "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
