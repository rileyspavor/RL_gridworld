"""Thread-safe runtime variant selection used by environment customization hooks.

The environment package calls into this module through ``coverage_gridworld.custom``.
It stores the currently active observation/reward variant names and parameters so
all environments created in a run use a consistent feature/reward definition.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import gymnasium as gym
import numpy as np

from project_rl.observations import get_observation_variant
from project_rl.rewards import compute_reward as compute_reward_variant
from project_rl.rewards import get_reward_variant


@dataclass
class CustomVariantState:
    """In-memory snapshot of the active observation and reward variants."""

    observation_name: str = "frontier_features"
    observation_params: dict[str, Any] = field(default_factory=dict)
    reward_name: str = "dense_coverage"
    reward_params: dict[str, Any] = field(default_factory=dict)


_STATE = CustomVariantState()
_LOCK = RLock()


def set_custom_variants(
    observation_name: str,
    reward_name: str,
    observation_params: dict[str, Any] | None = None,
    reward_params: dict[str, Any] | None = None,
) -> None:
    """Validate and install active observation/reward variants.

    Args:
        observation_name: Registered observation variant key.
        reward_name: Registered reward variant key.
        observation_params: Optional observation-variant parameters.
        reward_params: Optional reward-variant parameters.
    """
    observation_params = dict(observation_params or {})
    reward_params = dict(reward_params or {})

    get_observation_variant(observation_name)
    get_reward_variant(reward_name)

    with _LOCK:
        _STATE.observation_name = observation_name
        _STATE.observation_params = observation_params
        _STATE.reward_name = reward_name
        _STATE.reward_params = reward_params


def current_variants() -> dict[str, Any]:
    """Return a thread-safe copy of currently configured variants.

    Returns:
        Dictionary with ``observation`` and ``reward`` name/params entries.
    """
    with _LOCK:
        return {
            "observation": {
                "name": _STATE.observation_name,
                "params": dict(_STATE.observation_params),
            },
            "reward": {
                "name": _STATE.reward_name,
                "params": dict(_STATE.reward_params),
            },
        }


def _infer_env_from_stack() -> Any | None:
    """Best-effort discovery of the environment object from call frames.

    ``coverage_gridworld.custom.observation`` only receives the raw grid, so this
    helper walks stack frames to find the environment instance when available.

    Returns:
        Environment-like object exposing grid/agent fields, or ``None``.
    """
    frame = inspect.currentframe()
    try:
        while frame is not None:
            candidate = frame.f_locals.get("self")
            if candidate is not None and hasattr(candidate, "grid") and hasattr(candidate, "agent_pos"):
                return candidate
            frame = frame.f_back
    finally:
        del frame
    return None


def custom_observation_space(env: gym.Env) -> gym.spaces.Space:
    """Build the observation space for the active observation variant.

    Args:
        env: Environment instance used for space construction.

    Returns:
        Gymnasium space expected by the active observation transform.
    """
    with _LOCK:
        variant_name = _STATE.observation_name
        params = dict(_STATE.observation_params)
    space_fn, _ = get_observation_variant(variant_name)
    return space_fn(env, params)


def custom_observation(grid: np.ndarray) -> np.ndarray:
    """Transform a raw environment grid using the active observation variant.

    Args:
        grid: Raw 10x10x3 grid observation from the environment.

    Returns:
        Feature vector/array defined by the active observation transform.
    """
    env = _infer_env_from_stack()
    with _LOCK:
        variant_name = _STATE.observation_name
        params = dict(_STATE.observation_params)
    _, transform_fn = get_observation_variant(variant_name)
    return transform_fn(grid, env, params)


def custom_reward(info: dict[str, Any]) -> float:
    """Compute reward with the currently selected reward variant.

    Args:
        info: Environment ``info`` payload for the latest step.

    Returns:
        Scalar reward value.
    """
    with _LOCK:
        variant_name = _STATE.reward_name
        params = dict(_STATE.reward_params)
    return compute_reward_variant(variant_name, info, params)
