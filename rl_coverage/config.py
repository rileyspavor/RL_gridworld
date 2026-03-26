from __future__ import annotations

import copy
import json
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "ppo_frontier_dense",
        "output_root": "runs",
        "seed": 7,
        "device": "auto",
        "notes": "",
    },
    "environment": {
        "id": "sneaky_enemies",
        "map_suite": None,
        "render_mode": None,
        "activate_game_status": False,
    },
    "observation": {
        "name": "frontier_features",
        "params": {},
    },
    "reward": {
        "name": "dense_coverage",
        "params": {},
    },
    "algorithm": {
        "name": "PPO",
        "policy": "MlpPolicy",
        "kwargs": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 256,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
    },
    "training": {
        "total_timesteps": 300000,
        "n_envs": 4,
        "eval_freq": 25000,
        "eval_episodes": 12,
        "deterministic_eval": True,
        "save_best_model": True,
    },
    "evaluation": {
        "episodes": 20,
        "deterministic": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("rb") as handle:
        user_config = tomllib.load(handle)
    resolved = _deep_merge(DEFAULT_CONFIG, user_config)
    resolved["_config_path"] = str(path.resolve())
    return resolved


def resolve_run_dir(config: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)

    root = Path(config["experiment"]["output_root"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = config["experiment"]["name"]
    return root / f"{stamp}-{name}"


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
