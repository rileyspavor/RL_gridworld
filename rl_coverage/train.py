from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from rl_coverage.config import load_config, resolve_run_dir, save_json
from rl_coverage.env_factory import make_env_builder
from rl_coverage.metrics import evaluate_model, evaluation_text


def _constant_schedule(value: float):
    return lambda _: value


def _apply_loaded_algorithm_overrides(model, algorithm_kwargs: dict) -> None:
    scalar_attrs = {
        "gamma",
        "gae_lambda",
        "ent_coef",
        "vf_coef",
        "batch_size",
        "n_steps",
    }
    for key in scalar_attrs:
        if key in algorithm_kwargs:
            setattr(model, key, algorithm_kwargs[key])

    if "learning_rate" in algorithm_kwargs:
        learning_rate = float(algorithm_kwargs["learning_rate"])
        model.learning_rate = learning_rate
        model.lr_schedule = _constant_schedule(learning_rate)
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = learning_rate

    if "clip_range" in algorithm_kwargs and hasattr(model, "clip_range"):
        model.clip_range = _constant_schedule(float(algorithm_kwargs["clip_range"]))

    if "clip_range_vf" in algorithm_kwargs and hasattr(model, "clip_range_vf"):
        clip_range_vf = algorithm_kwargs["clip_range_vf"]
        model.clip_range_vf = None if clip_range_vf is None else _constant_schedule(float(clip_range_vf))


def _algorithm_registry():
    try:
        from stable_baselines3 import A2C, DQN, PPO
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed. Install the training dependencies first, "
            "then rerun this command."
        ) from exc
    return {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent for Coverage Gridworld.")
    parser.add_argument("--config", required=True, help="Path to a TOML experiment config.")
    parser.add_argument("--output-dir", default=None, help="Optional override for the run directory.")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for the experiment seed.")
    return parser.parse_args(argv)


def _progress_bar_available() -> bool:
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401
    except ImportError:
        return False
    return True


def _transfer_policy_prefix(source_model, target_model, prefix_features: int) -> None:
    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    updated_state = dict(target_state)
    copied_keys: list[str] = []

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            continue
        if source_tensor.shape == target_tensor.shape:
            updated_state[key] = source_tensor.clone()
            copied_keys.append(key)
            continue
        if (
            prefix_features > 0
            and source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[1] == prefix_features
            and target_tensor.shape[1] >= prefix_features
        ):
            merged = target_tensor.clone()
            merged[:, :prefix_features] = source_tensor
            merged[:, prefix_features:] = 0.0
            updated_state[key] = merged
            copied_keys.append(key)

    target_model.policy.load_state_dict(updated_state)
    print(
        f"[train] Transferred {len(copied_keys)} policy tensors from {source_model.__class__.__name__} "
        f"with observation prefix {prefix_features}."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed

    run_dir = resolve_run_dir(config, output_dir=args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)

    algorithms = _algorithm_registry()
    algo_name = config["algorithm"]["name"]
    if algo_name not in algorithms:
        valid = ", ".join(sorted(algorithms))
        raise SystemExit(f"Unknown algorithm '{algo_name}'. Valid algorithms: {valid}")

    try:
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise SystemExit("stable-baselines3 common utilities are unavailable.") from exc

    train_builder = make_env_builder(config)
    eval_builder = make_env_builder(config)
    vec_env = make_vec_env(train_builder, n_envs=int(config["training"]["n_envs"]), seed=seed)

    policy = config["algorithm"]["policy"]
    algorithm_kwargs = dict(config["algorithm"].get("kwargs", {}))
    algorithm_kwargs.setdefault("seed", seed)
    algorithm_kwargs.setdefault("verbose", 1)
    algorithm_kwargs.setdefault("device", config["experiment"].get("device", "auto"))
    algorithm_kwargs.setdefault("tensorboard_log", str(run_dir / "tensorboard"))

    model_cls = algorithms[algo_name]
    init_model_path = config["algorithm"].get("init_model_path")
    transfer_model_path = config["algorithm"].get("transfer_model_path")
    transfer_observation_prefix = int(config["algorithm"].get("transfer_observation_prefix", 0))
    if init_model_path:
        model = model_cls.load(init_model_path, env=vec_env, device=algorithm_kwargs.get("device", "auto"))
        model.tensorboard_log = str(run_dir / "tensorboard")
        _apply_loaded_algorithm_overrides(model, algorithm_kwargs)
        print(f"[train] Initialized from existing model: {Path(init_model_path).resolve()}")
        if "learning_rate" in algorithm_kwargs:
            print(f"[train] Override learning_rate -> {float(algorithm_kwargs['learning_rate']):g}")
    else:
        model = model_cls(policy, vec_env, **algorithm_kwargs)
        if transfer_model_path:
            source_model = model_cls.load(transfer_model_path, device=algorithm_kwargs.get("device", "auto"))
            _transfer_policy_prefix(source_model, model, prefix_features=transfer_observation_prefix)
            del source_model
            print(f"[train] Warm-started from policy transfer: {Path(transfer_model_path).resolve()}")

    from rl_coverage.callbacks import PeriodicEvalCallback

    callback = PeriodicEvalCallback(
        env_builder=eval_builder,
        eval_episodes=int(config["training"]["eval_episodes"]),
        eval_freq=int(config["training"]["eval_freq"]),
        deterministic=bool(config["training"]["deterministic_eval"]),
        output_dir=run_dir,
        save_best_model=bool(config["training"].get("save_best_model", True)),
    )

    total_timesteps = int(config["training"]["total_timesteps"])
    reset_num_timesteps = bool(config["training"].get("reset_num_timesteps", init_model_path is None))
    progress_bar = _progress_bar_available()
    if not progress_bar:
        print("[train] rich/tqdm not available; continuing without progress bar.")
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
        deterministic=bool(config["evaluation"]["deterministic"]),
        seed=seed,
    )
    save_json(run_dir / "final_evaluation.json", final_evaluation)

    summary = {
        "run_dir": str(Path(run_dir).resolve()),
        "config_path": config["_config_path"],
        "algorithm": algo_name,
        "observation": config["observation"]["name"],
        "reward": config["reward"]["name"],
        "environment": config["environment"]["id"],
        "map_suite": config["environment"].get("map_suite"),
        "total_timesteps": total_timesteps,
        **final_evaluation["summary"],
    }
    save_json(run_dir / "summary.json", summary)
    print(f"[final] {evaluation_text(final_evaluation['summary'])}")
    print(f"Artifacts saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
