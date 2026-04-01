"""Render trained PPO policies for qualitative inspection.

Unlike ``evaluate.py``, this script opens the environment in human-render mode and
steps through episodes in real time. It is mainly used to inspect behavior across
maps after training and to sanity-check whether policy decisions look sensible.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from project_rl.config import load_config, load_json
from project_rl.env_factory import make_env
from project_rl.maps import MAP_SUITES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for interactive playback.

    Args:
        argv: Optional argument list for testing or embedding.

    Returns:
        Parsed playback options such as run source, episode count, and speed.
    """
    parser = argparse.ArgumentParser(description="Render a trained PPO agent playing Coverage Gridworld.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing config.json and model files.")
    parser.add_argument("--config", default=None, help="Config file path when not using --run-dir.")
    parser.add_argument("--model", default=None, help="Model path (zip or basename without .zip).")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of rendered episodes. Defaults to one pass over the configured map suite, or 3 if no suite is set.",
    )
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay between rendered steps.")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions instead of deterministic.")
    return parser.parse_args(argv)


def _resolve_model_path(run_dir: Path, explicit_model: str | None) -> Path:
    """Select the checkpoint to load from a run directory.

    Args:
        run_dir: Training run directory containing model artifacts.
        explicit_model: Optional direct override from CLI.

    Returns:
        Chosen checkpoint path, preferring ``best_model`` over ``final_model``.
    """
    if explicit_model is not None:
        return Path(explicit_model)
    if (run_dir / "best_model.zip").exists():
        return run_dir / "best_model"
    return run_dir / "final_model"


def _default_episode_count(config: dict) -> int:
    """Derive a sensible default number of rendered episodes.

    If the config uses a named map suite, one episode is rendered per map so the
    visual pass covers the whole suite once. Otherwise it falls back to ``3``.

    Args:
        config: Full experiment configuration dictionary.

    Returns:
        Default episode count for playback.
    """
    map_suite = config.get("environment", {}).get("map_suite")
    if map_suite and map_suite in MAP_SUITES:
        return len(MAP_SUITES[map_suite])
    return 3


def main(argv: list[str] | None = None) -> int:
    """Load a trained model, render episodes, and close the environment cleanly.

    Args:
        argv: Optional argument list.

    Returns:
        ``0`` when playback completes.
    """
    args = parse_args(argv)
    if args.run_dir is None and (args.config is None or args.model is None):
        raise SystemExit("Use --run-dir, or provide both --config and --model.")

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is not None:
        config = load_json(run_dir / "config.json")
        model_path = _resolve_model_path(run_dir, args.model)
    else:
        config = load_config(args.config)
        model_path = Path(args.model)

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("stable-baselines3 is required to run trained models.") from exc

    config["environment"]["render_mode"] = "human"
    config["environment"]["activate_game_status"] = True
    env = make_env(config, render_mode="human")
    model = PPO.load(model_path)
    episode_count = int(args.episodes) if args.episodes is not None else _default_episode_count(config)

    try:
        for episode_index in range(episode_count):
            print(f"[play] Episode {episode_index + 1}/{episode_count}")
            observation, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=not args.stochastic)
                observation, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                if args.sleep > 0:
                    time.sleep(args.sleep)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
