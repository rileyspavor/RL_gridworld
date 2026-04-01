"""Assignment-approved customization entrypoints for Coverage Gridworld.

The course environment imports this file directly. To keep `env.py` untouched while
still supporting multiple observation spaces and reward functions, these hooks act
as a thin bridge into the configurable implementations under ``project_rl``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_rl.customization import (
    custom_observation,
    custom_observation_space,
    custom_reward,
    set_custom_variants,
)


def configure_variants(
    observation_name: str,
    reward_name: str,
    observation_params: dict[str, Any] | None = None,
    reward_params: dict[str, Any] | None = None,
) -> None:
    """Install the active observation/reward definitions for future env calls.

    This helper is the submission-safe way to switch between custom observation
    spaces and reward functions without editing the environment implementation.
    """
    set_custom_variants(
        observation_name=observation_name,
        reward_name=reward_name,
        observation_params=observation_params,
        reward_params=reward_params,
    )


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """Return the Gymnasium observation space for the active feature variant."""
    return custom_observation_space(env)


def observation(grid: np.ndarray):
    """Transform the raw grid observation into the active custom observation."""
    return custom_observation(grid)


def reward(info: dict) -> float:
    """Compute the scalar reward for the current step using the active reward."""
    return custom_reward(info)


def set_custom_variants_from_config(config: dict[str, Any]) -> None:
    """Apply observation/reward settings from a resolved experiment config."""
    configure_variants(
        observation_name=config["observation"]["name"],
        reward_name=config["reward"]["name"],
        observation_params=config["observation"].get("params", {}),
        reward_params=config["reward"].get("params", {}),
    )
