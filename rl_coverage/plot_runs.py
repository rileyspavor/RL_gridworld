from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CSV summaries and plots from Coverage Gridworld runs.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run folders.")
    parser.add_argument("--output-dir", default="results/plots", help="Directory for generated CSV/PNG artifacts.")
    return parser.parse_args(argv)


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text())
        evaluations_path = run_dir / "evaluations.json"
        evaluations = []
        if evaluations_path.exists():
            evaluations = json.loads(evaluations_path.read_text()).get("items", [])
        summary["_run_name"] = run_dir.name
        summary["_run_dir"] = str(run_dir)
        summary["_evaluations"] = evaluations
        runs.append(summary)
    return runs


def write_leaderboard_csv(runs: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "leaderboard.csv"
    fieldnames = [
        "run_name",
        "algorithm",
        "observation",
        "reward",
        "environment",
        "map_suite",
        "total_timesteps",
        "mean_coverage",
        "success_rate",
        "death_rate",
        "timeout_rate",
        "mean_reward",
        "mean_length",
        "config_path",
        "run_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(runs, key=lambda row: (row.get("mean_coverage", 0.0), row.get("success_rate", 0.0)), reverse=True):
            writer.writerow(
                {
                    "run_name": item["_run_name"],
                    "algorithm": item.get("algorithm"),
                    "observation": item.get("observation"),
                    "reward": item.get("reward"),
                    "environment": item.get("environment"),
                    "map_suite": item.get("map_suite"),
                    "total_timesteps": item.get("total_timesteps"),
                    "mean_coverage": item.get("mean_coverage"),
                    "success_rate": item.get("success_rate"),
                    "death_rate": item.get("death_rate"),
                    "timeout_rate": item.get("timeout_rate"),
                    "mean_reward": item.get("mean_reward"),
                    "mean_length": item.get("mean_length"),
                    "config_path": item.get("config_path"),
                    "run_dir": item.get("run_dir"),
                }
            )
    return path


def _group_key(run: dict[str, Any]) -> str:
    return f"{run.get('observation')}\n{run.get('reward')}"


def plot_metric_bars(runs: list[dict[str, Any]], metric: str, ylabel: str, output_path: Path) -> None:
    ordered = sorted(runs, key=lambda row: row.get(metric, 0.0), reverse=True)
    labels = [_group_key(run) for run in ordered]
    values = [float(run.get(metric, 0.0)) for run in ordered]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(ordered)), values, color="#4C78A8")
    plt.xticks(range(len(ordered)), labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by observation/reward setup")
    plt.tight_layout()

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_learning_curves(runs: list[dict[str, Any]], output_path: Path) -> None:
    plt.figure(figsize=(12, 7))
    any_curve = False
    for run in runs:
        evaluations = run.get("_evaluations", [])
        if not evaluations:
            continue
        xs = [item["timesteps"] for item in evaluations]
        ys = [item["summary"]["mean_coverage"] for item in evaluations]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=_group_key(run))
        any_curve = True

    if not any_curve:
        plt.close()
        return

    plt.xlabel("Training timesteps")
    plt.ylabel("Evaluation mean coverage")
    plt.title("Learning curves")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def write_markdown_summary(runs: list[dict[str, Any]], output_dir: Path) -> Path:
    ordered = sorted(runs, key=lambda row: (row.get("mean_coverage", 0.0), row.get("success_rate", 0.0)), reverse=True)
    lines = [
        "# Coverage Gridworld Run Summary",
        "",
        f"Runs analyzed: {len(ordered)}",
        "",
        "## Leaderboard",
        "",
    ]
    for index, run in enumerate(ordered, start=1):
        lines.append(
            f"{index}. `{run['_run_name']}` — obs={run.get('observation')}, reward={run.get('reward')}, "
            f"coverage={run.get('mean_coverage', 0.0):.3f}, success={run.get('success_rate', 0.0):.3f}, "
            f"death={run.get('death_rate', 0.0):.3f}, reward={run.get('mean_reward', 0.0):.3f}"
        )
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    runs = load_runs(runs_dir)
    if not runs:
        raise SystemExit("No completed runs found.")

    leaderboard_csv = write_leaderboard_csv(runs, output_dir)
    plot_metric_bars(runs, metric="mean_coverage", ylabel="Mean coverage", output_path=output_dir / "mean_coverage_by_combo.png")
    plot_metric_bars(runs, metric="success_rate", ylabel="Success rate", output_path=output_dir / "success_rate_by_combo.png")
    plot_metric_bars(runs, metric="mean_reward", ylabel="Mean reward", output_path=output_dir / "mean_reward_by_combo.png")
    plot_learning_curves(runs, output_path=output_dir / "learning_curves.png")
    summary_md = write_markdown_summary(runs, output_dir)

    print(f"Wrote {leaderboard_csv}")
    print(f"Wrote {summary_md}")
    print(f"Wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
