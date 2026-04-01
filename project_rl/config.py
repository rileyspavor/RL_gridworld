"""Configuration and artifact serialization utilities for experiments.

The training pipeline starts from a TOML file, merges it with project defaults,
and writes resolved JSON artifacts into run directories for reproducibility.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


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
        "map_suite": "all_standard",
        "render_mode": "",
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
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 256,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
    },
    "training": {
        "total_timesteps": 150000,
        "n_envs": 4,
        "eval_freq": 20000,
        "eval_episodes": 10,
        "deterministic_eval": True,
        "save_best_model": True,
        "reset_num_timesteps": True,
    },
    "evaluation": {
        "episodes": 20,
        "deterministic": True,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` values into a deep copy of ``base``.

    Args:
        base: Default configuration tree.
        override: User-specified configuration values.

    Returns:
        New merged dictionary without mutating either input.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a TOML experiment config and merge it with defaults.

    Args:
        config_path: Path to a user TOML configuration file.

    Returns:
        Fully resolved config dictionary including ``_config_path`` metadata.
    """
    path = Path(config_path)
    with path.open("rb") as handle:
        user_config = tomllib.load(handle)
    resolved = deep_merge(DEFAULT_CONFIG, user_config)
    resolved["_config_path"] = str(path.resolve())
    return resolved


def resolve_run_dir(config: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    """Determine where run artifacts should be written.

    Args:
        config: Resolved experiment config.
        output_dir: Optional explicit output directory override.

    Returns:
        Concrete run directory path.
    """
    if output_dir is not None:
        return Path(output_dir)

    root = Path(config["experiment"]["output_root"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"{timestamp}-{config['experiment']['name']}"


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON to disk with stable formatting and parent creation.

    Args:
        path: Destination file path.
        payload: JSON-serializable dictionary to persist.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        path: Source JSON file path.

    Returns:
        Parsed JSON payload.
    """
    return json.loads(Path(path).read_text())
