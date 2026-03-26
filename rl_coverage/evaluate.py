from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_coverage.config import load_config, save_json
from rl_coverage.env_factory import make_env_builder
from rl_coverage.metrics import evaluate_model, evaluation_text


def _algorithm_registry():
    try:
        from stable_baselines3 import A2C, DQN, PPO
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed. Install the training dependencies first, then rerun this command."
        ) from exc
    return {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Coverage Gridworld model.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing config.json and model files.")
    parser.add_argument("--config", default=None, help="Config path when evaluating outside a run directory.")
    parser.add_argument("--model", default=None, help="Path to a model zip/basename. Defaults to best_model or final_model in the run dir.")
    parser.add_argument("--episodes", type=int, default=None, help="Override evaluation episode count.")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic policy evaluation.")
    return parser.parse_args(argv)


def _load_config_from_run_dir(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    return json.loads(config_path.read_text())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.run_dir is None and args.config is None:
        raise SystemExit("Provide either --run-dir or --config.")

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is not None:
        config = _load_config_from_run_dir(run_dir)
    else:
        config = load_config(args.config)

    algorithm_name = config["algorithm"]["name"]
    algorithms = _algorithm_registry()
    if algorithm_name not in algorithms:
        valid = ", ".join(sorted(algorithms))
        raise SystemExit(f"Unknown algorithm '{algorithm_name}'. Valid algorithms: {valid}")

    model_path = args.model
    if model_path is None:
        if run_dir is None:
            raise SystemExit("When not using --run-dir you must also provide --model.")
        if (run_dir / "best_model.zip").exists():
            model_path = run_dir / "best_model"
        else:
            model_path = run_dir / "final_model"
    else:
        model_path = Path(model_path)

    env_builder = make_env_builder(config)
    model_cls = algorithms[algorithm_name]
    model = model_cls.load(model_path)

    evaluation = evaluate_model(
        model,
        env_builder=env_builder,
        episodes=args.episodes or int(config["evaluation"]["episodes"]),
        deterministic=args.deterministic or bool(config["evaluation"].get("deterministic", True)),
    )
    summary = evaluation["summary"]
    print(evaluation_text(summary))

    if run_dir is not None:
        save_json(run_dir / "manual_evaluation.json", evaluation)
        save_json(run_dir / "manual_evaluation_summary.json", summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
