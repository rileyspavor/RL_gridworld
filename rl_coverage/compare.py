from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed Coverage Gridworld runs.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run subfolders with summary.json files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    summaries = []

    if not runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        payload["_summary_path"] = str(summary_path)
        summaries.append(payload)

    if not summaries:
        print("No completed runs found.")
        return 0

    summaries.sort(key=lambda item: (item.get("mean_coverage", 0.0), item.get("success_rate", 0.0)), reverse=True)

    header = (
        f"{'run':40} {'algo':6} {'obs':18} {'reward':18} {'coverage':>9} {'success':>8} {'death':>8} {'reward_mean':>12}"
    )
    print(header)
    print("-" * len(header))

    for item in summaries:
        run_name = Path(item["run_dir"]).name
        print(
            f"{run_name:40} "
            f"{item['algorithm']:6} "
            f"{item['observation'][:18]:18} "
            f"{item['reward'][:18]:18} "
            f"{item['mean_coverage']:9.3f} "
            f"{item['success_rate']:8.3f} "
            f"{item['death_rate']:8.3f} "
            f"{item['mean_reward']:12.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
