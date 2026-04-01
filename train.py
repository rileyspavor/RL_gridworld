"""Train one Coverage Gridworld PPO experiment from a TOML config.

This CLI wraps :func:`project_rl.training.train_experiment` so a single run can be
launched from the command line with optional overrides (seed, output location, and
timesteps) while keeping the experiment definition config-driven.
"""

from __future__ import annotations

import argparse

from project_rl.config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for a single training run.

    Args:
        argv: Optional argument list used mainly for tests.

    Returns:
        Parsed argparse namespace containing config and override options.
    """
    parser = argparse.ArgumentParser(description="Train a PPO agent for Coverage Gridworld.")
    parser.add_argument("--config", required=True, help="Path to a TOML config file.")
    parser.add_argument("--output-dir", default=None, help="Optional explicit run directory.")
    parser.add_argument("--seed", type=int, default=None, help="Override config experiment seed.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override training.total_timesteps for this run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load configuration, run training, and return a process exit code.

    Args:
        argv: Optional argument list for programmatic invocation.

    Returns:
        ``0`` after training artifacts have been written.
    """
    args = parse_args(argv)
    config = load_config(args.config)
    from project_rl.training import train_experiment

    train_experiment(
        config,
        output_dir=args.output_dir,
        seed_override=args.seed,
        total_timesteps_override=args.timesteps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
