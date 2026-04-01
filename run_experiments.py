"""Run the full assignment experiment matrix and optional continuation training.

This script automates the 2x3 observation/reward sweep, writes a run manifest,
optionally trains a continuation model from the top run, and can generate plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from project_rl.config import load_config, load_json, save_json
from project_rl.plotting import generate_experiment_report

MATRIX_CONFIG_FILES = [
    "frontier_features_sparse_coverage.toml",
    "frontier_features_dense_coverage.toml",
    "frontier_features_survival_coverage.toml",
    "temporal_frontier_features_sparse_coverage.toml",
    "temporal_frontier_features_dense_coverage.toml",
    "temporal_frontier_features_survival_coverage.toml",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for matrix experiments.

    Args:
        argv: Optional argument list for tests or embedded invocation.

    Returns:
        Parsed matrix execution settings and optional continuation options.
    """
    parser = argparse.ArgumentParser(description="Run the 2x3 observation/reward PPO experiment matrix.")
    parser.add_argument("--config-dir", default="configs/experiments", help="Directory containing matrix TOML files.")
    parser.add_argument("--output-root", default="runs/matrix", help="Root folder for experiment runs.")
    parser.add_argument("--report-dir", default="results/plots", help="Directory for generated comparison plots.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps for every matrix run.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed for every matrix run.")
    parser.add_argument("--train-best", action="store_true", help="Continue training from the best matrix run.")
    parser.add_argument(
        "--best-timesteps",
        type=int,
        default=350000,
        help="Timesteps for optional best-agent continuation training.",
    )
    parser.add_argument(
        "--best-output-root",
        default="runs/best_agent",
        help="Root folder for optional best-agent continuation training.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot/report generation.")
    return parser.parse_args(argv)


def _matrix_config_paths(config_dir: Path) -> list[Path]:
    """Resolve and validate the required matrix configuration files.

    Args:
        config_dir: Directory expected to contain all matrix TOML files.

    Returns:
        Ordered list of required config file paths.

    Raises:
        SystemExit: If any required file is missing.
    """
    paths = [config_dir / name for name in MATRIX_CONFIG_FILES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {item}" for item in missing)
        raise SystemExit(f"Missing required matrix configs:\n{missing_text}")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Execute matrix training runs and produce report artifacts.

    Args:
        argv: Optional argument list.

    Returns:
        ``0`` when all requested runs/reporting complete.
    """
    args = parse_args(argv)
    from project_rl.training import train_experiment

    config_paths = _matrix_config_paths(Path(args.config_dir))
    matrix_output_root = Path(args.output_root)
    matrix_output_root.mkdir(parents=True, exist_ok=True)

    run_dirs: list[Path] = []
    run_summaries: list[dict] = []

    for config_path in config_paths:
        config = load_config(config_path)
        config["experiment"]["output_root"] = str(matrix_output_root)
        config["experiment"]["name"] = config_path.stem

        run_dir, summary = train_experiment(
            config,
            seed_override=args.seed,
            total_timesteps_override=args.timesteps,
        )
        run_dirs.append(run_dir)
        run_summaries.append(summary)

    leaderboard = sorted(run_summaries, key=lambda item: item.get("mean_coverage", 0.0), reverse=True)
    manifest = {
        "runs": [str(run_dir.resolve()) for run_dir in run_dirs],
        "leaderboard": leaderboard,
    }
    save_json(matrix_output_root / "latest_matrix_runs.json", manifest)

    best_agent_run_dir = None
    if args.train_best and leaderboard:
        best_summary = leaderboard[0]
        best_run_dir = Path(best_summary["run_dir"])
        best_config = load_json(best_run_dir / "config.json")
        best_config["experiment"]["output_root"] = str(args.best_output_root)
        best_config["experiment"]["name"] = "best_agent_continuation"
        best_config["algorithm"]["init_model_path"] = str(
            best_run_dir / ("best_model" if (best_run_dir / "best_model.zip").exists() else "final_model")
        )
        best_config["training"]["total_timesteps"] = int(args.best_timesteps)
        best_config["training"]["reset_num_timesteps"] = False

        best_agent_run_dir, _ = train_experiment(best_config)
        manifest["best_agent_run"] = str(best_agent_run_dir.resolve())
        save_json(matrix_output_root / "latest_matrix_runs.json", manifest)

    if not args.skip_plots:
        report_files = generate_experiment_report(run_dirs, output_dir=args.report_dir)
        print("[report] Generated files:")
        for report_file in report_files:
            print(f"- {report_file}")

    print("[matrix] Top result:")
    if leaderboard:
        top = leaderboard[0]
        print(
            f"- obs={top['observation']} reward={top['reward']} "
            f"mean_coverage={top['mean_coverage']:.3f} success_rate={top['success_rate']:.3f}"
        )

    if best_agent_run_dir is not None:
        print(f"[best-agent] Continuation run: {best_agent_run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
