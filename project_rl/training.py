"""High-level training orchestration for Coverage Gridworld PPO experiments.

This module is where configuration turns into actual Stable-Baselines training.
It is responsible for:
- creating run directories and saving resolved configs
- building vectorized train/eval environments
- constructing a PPO model or continuing from a checkpoint
- running periodic evaluation callbacks
- writing final summaries and model artifacts
"""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from project_rl.callbacks import PeriodicEvalCallback
from project_rl.config import resolve_run_dir, save_json
from project_rl.env_factory import make_env_builder
from project_rl.metrics import evaluate_model, evaluation_text


def _algorithm_registry():
    """Return the assignment's supported RL algorithms.

    The current project intentionally supports PPO only, but keeping the lookup in
    one place makes the CLI and training flow easier to understand.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "stable-baselines3 is not installed. Install training dependencies and retry."
        ) from exc
    return {"PPO": PPO}


def _constant_schedule(value: float):
    """Wrap a scalar as a Stable-Baselines schedule callable."""
    return lambda _: value


def _apply_loaded_algorithm_overrides(model, algorithm_kwargs: dict[str, Any]) -> None:
    """Reapply selected PPO hyperparameters after loading a checkpoint.

    This is mainly needed for continuation training, where a loaded model should
    obey the new config's learning-rate and PPO-scalar overrides.
    """
    for key in ("gamma", "gae_lambda", "ent_coef", "vf_coef", "batch_size", "n_steps"):
        if key in algorithm_kwargs:
            setattr(model, key, algorithm_kwargs[key])

    if "learning_rate" in algorithm_kwargs:
        learning_rate = float(algorithm_kwargs["learning_rate"])
        model.learning_rate = learning_rate
        model.lr_schedule = _constant_schedule(learning_rate)
        if hasattr(model.policy, "optimizer"):
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = learning_rate

    if "clip_range" in algorithm_kwargs:
        model.clip_range = _constant_schedule(float(algorithm_kwargs["clip_range"]))


def _progress_bar_available() -> bool:
    """Check whether optional progress-bar dependencies are installed."""
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401
    except ImportError:
        return False
    return True


def _build_eval_schedule(training_config: dict[str, Any]) -> list[int] | None:
    """Build an optional evaluation schedule from training config settings.

    Supported simple mode:
    - ``eval_points`` + ``eval_spacing = 'log'``

    Falls back to ``None`` when no custom schedule is requested, allowing the
    existing fixed ``eval_freq`` behavior to remain unchanged.
    """
    eval_points = training_config.get("eval_points")
    eval_spacing = str(training_config.get("eval_spacing", "")).strip().lower()
    total_timesteps = int(training_config["total_timesteps"])

    if eval_points is None:
        return None

    eval_points = int(eval_points)
    if eval_points <= 0:
        return None

    if eval_spacing != "log":
        raise SystemExit("training.eval_spacing currently supports only 'log'.")

    if total_timesteps <= 1:
        return [max(1, total_timesteps)]

    raw_values = np.geomspace(1, total_timesteps, num=eval_points)
    schedule = sorted({max(1, min(total_timesteps, int(round(value)))) for value in raw_values})
    if schedule[-1] != total_timesteps:
        schedule.append(total_timesteps)
    return schedule


def _prepare_model(config: dict[str, Any], vec_env, run_dir: Path):
    """Create a fresh PPO model or resume training from an existing checkpoint.

    Args:
        config: Resolved experiment configuration.
        vec_env: Vectorized training environment.
        run_dir: Current run directory used for tensorboard/log outputs.

    Returns:
        Configured Stable-Baselines PPO model.
    """
    algorithms = _algorithm_registry()
    algorithm_name = str(config["algorithm"]["name"]).upper()
    if algorithm_name != "PPO":
        raise SystemExit("This assignment framework supports PPO only.")

    model_cls = algorithms[algorithm_name]
    policy = str(config["algorithm"]["policy"])
    algorithm_kwargs = dict(config["algorithm"].get("kwargs", {}))
    seed = int(config["experiment"]["seed"])

    algorithm_kwargs.setdefault("seed", seed)
    algorithm_kwargs.setdefault("verbose", 1)
    algorithm_kwargs.setdefault("device", config["experiment"].get("device", "auto"))
    algorithm_kwargs.setdefault("tensorboard_log", str(run_dir / "tensorboard"))

    init_model_path = config["algorithm"].get("init_model_path")
    if init_model_path:
        model = model_cls.load(init_model_path, env=vec_env, device=algorithm_kwargs.get("device", "auto"))
        model.tensorboard_log = str(run_dir / "tensorboard")
        _apply_loaded_algorithm_overrides(model, algorithm_kwargs)
        print(f"[train] Continuing from {Path(init_model_path).resolve()}")
        return model

    return model_cls(policy, vec_env, **algorithm_kwargs)


def train_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    seed_override: int | None = None,
    total_timesteps_override: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Train one PPO experiment and write its artifacts to disk.

    Args:
        config: Resolved experiment configuration dictionary.
        output_dir: Optional explicit run directory override.
        seed_override: Optional seed override used by the CLI.
        total_timesteps_override: Optional total-timestep override used by the CLI.

    Returns:
        Tuple ``(run_dir, final_summary)`` describing the finished experiment.
    """
    config = copy.deepcopy(config)

    if seed_override is not None:
        config["experiment"]["seed"] = int(seed_override)
    if total_timesteps_override is not None:
        config["training"]["total_timesteps"] = int(total_timesteps_override)

    run_dir = resolve_run_dir(config, output_dir=output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)

    try:
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("stable-baselines3 common utilities are unavailable.") from exc

    train_builder = make_env_builder(config)
    eval_builder = make_env_builder(config)
    vec_env = make_vec_env(train_builder, n_envs=int(config["training"]["n_envs"]), seed=seed)

    model = _prepare_model(config, vec_env, run_dir)
    eval_schedule = _build_eval_schedule(config["training"])

    callback = PeriodicEvalCallback(
        env_builder=eval_builder,
        eval_episodes=int(config["training"]["eval_episodes"]),
        eval_freq=int(config["training"].get("eval_freq", 0)),
        eval_schedule=eval_schedule,
        deterministic=bool(config["training"]["deterministic_eval"]),
        output_dir=run_dir,
        save_best_model=bool(config["training"].get("save_best_model", True)),
    )

    progress_bar = _progress_bar_available()
    if not progress_bar:
        print("[train] rich/tqdm not available; training without progress bar.")

    total_timesteps = int(config["training"]["total_timesteps"])
    reset_num_timesteps = bool(config["training"].get("reset_num_timesteps", True))

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=progress_bar,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(run_dir / "final_model")
    vec_env.close()

    final_evaluation = evaluate_model(
        model,
        env_builder=eval_builder,
        episodes=int(config["evaluation"]["episodes"]),
        deterministic=bool(config["evaluation"].get("deterministic", True)),
        seed=seed,
    )
    save_json(run_dir / "final_evaluation.json", final_evaluation)

    summary = {
        "run_dir": str(run_dir.resolve()),
        "config_path": config.get("_config_path"),
        "algorithm": str(config["algorithm"]["name"]).upper(),
        "observation": config["observation"]["name"],
        "reward": config["reward"]["name"],
        "environment": config["environment"]["id"],
        "map_suite": config["environment"].get("map_suite"),
        "total_timesteps": total_timesteps,
        **final_evaluation["summary"],
    }
    save_json(run_dir / "summary.json", summary)

    print(f"[final] {evaluation_text(final_evaluation['summary'])}")
    print(f"[final] Artifacts saved to {run_dir}")
    return run_dir, summary


def evaluate_trained_model(
    config: dict[str, Any],
    model_path: str | Path,
    episodes: int | None = None,
    deterministic: bool | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    algorithms = _algorithm_registry()
    algorithm_name = str(config["algorithm"]["name"]).upper()
    if algorithm_name != "PPO":
        raise SystemExit("This assignment framework supports PPO only.")

    model_cls = algorithms[algorithm_name]
    model = model_cls.load(model_path)
    env_builder = make_env_builder(config)

    return evaluate_model(
        model,
        env_builder=env_builder,
        episodes=int(episodes or config["evaluation"]["episodes"]),
        deterministic=bool(config["evaluation"].get("deterministic", True) if deterministic is None else deterministic),
        seed=seed,
    )
