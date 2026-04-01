"""Evaluate a trained PPO checkpoint and print/save evaluation metrics.

This CLI supports two common workflows used in the project:
1) point to an existing run directory and reuse its stored config, or
2) pass a standalone config plus explicit model path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from project_rl.config import load_config, load_json, save_json
from project_rl.metrics import evaluation_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for checkpoint evaluation.

    Args:
        argv: Optional raw arguments for tests or embedded usage.

    Returns:
        Parsed options describing config source, model path, and evaluation mode.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained Coverage Gridworld PPO model.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing config.json and model files.")
    parser.add_argument("--config", default=None, help="Config path when not using --run-dir.")
    parser.add_argument("--model", default=None, help="Path to model zip or basename (without .zip).")
    parser.add_argument("--episodes", type=int, default=None, help="Override evaluation episode count.")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic policy evaluation.")
    return parser.parse_args(argv)


def _resolve_model_path(run_dir: Path, explicit_model: str | None) -> Path:
    """Resolve which model checkpoint path should be loaded for evaluation.

    The resolver prefers an explicit user-provided path. Otherwise it uses
    ``best_model`` when present and falls back to ``final_model``.

    Args:
        run_dir: Directory containing run artifacts.
        explicit_model: Optional direct checkpoint override.

    Returns:
        Filesystem path (zip basename allowed by Stable-Baselines3).
    """
    if explicit_model is not None:
        return Path(explicit_model)

    if (run_dir / "best_model.zip").exists():
        return run_dir / "best_model"
    return run_dir / "final_model"


def main(argv: list[str] | None = None) -> int:
    """Run model evaluation and optionally persist manual evaluation outputs.

    Args:
        argv: Optional command-line argument list.

    Returns:
        ``0`` when evaluation completes successfully.
    """
    args = parse_args(argv)
    if args.run_dir is None and args.config is None:
        raise SystemExit("Provide either --run-dir or --config.")

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is not None:
        config = load_json(run_dir / "config.json")
        model_path = _resolve_model_path(run_dir, args.model)
    else:
        config = load_config(args.config)
        if args.model is None:
            raise SystemExit("When using --config directly, provide --model.")
        model_path = Path(args.model)

    from project_rl.training import evaluate_trained_model

    evaluation = evaluate_trained_model(
        config,
        model_path=model_path,
        episodes=args.episodes,
        deterministic=True if args.deterministic else None,
    )
    print(evaluation_text(evaluation["summary"]))

    if run_dir is not None:
        save_json(run_dir / "manual_evaluation.json", evaluation)
        save_json(run_dir / "manual_evaluation_summary.json", evaluation["summary"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
