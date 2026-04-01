"""Environment construction helpers shared by training, evaluation, and playback.

This module applies config-selected observation/reward variants and map suites,
then creates Gymnasium environments with consistent wrapper behavior.
"""

from __future__ import annotations

from typing import Any

from project_rl.bootstrap import ensure_env_package_path

ensure_env_package_path()

import gymnasium as gym
import coverage_gridworld  # noqa: F401

from coverage_gridworld.custom import set_custom_variants

from project_rl.maps import resolve_map_suite


def _coerce_render_mode(value: str | None) -> str | None:
    """Normalize render-mode strings from configuration values.

    Args:
        value: Raw render-mode string or ``None``.

    Returns:
        Trimmed render mode, or ``None`` when empty.
    """
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def make_env(config: dict[str, Any], render_mode: str | None = None) -> gym.Env:
    """Create a configured Coverage Gridworld environment instance.

    Args:
        config: Resolved experiment configuration dictionary.
        render_mode: Optional explicit render mode override.

    Returns:
        Gymnasium environment wrapped with ``RecordEpisodeStatistics``.
    """
    env_config = config["environment"]
    map_suite = resolve_map_suite(env_config.get("map_suite"))
    env_id = env_config["id"] if map_suite is None else "standard"

    set_custom_variants(
        observation_name=config["observation"]["name"],
        reward_name=config["reward"]["name"],
        observation_params=config["observation"].get("params", {}),
        reward_params=config["reward"].get("params", {}),
    )

    chosen_render_mode = render_mode
    if chosen_render_mode is None:
        chosen_render_mode = _coerce_render_mode(env_config.get("render_mode"))

    env = gym.make(
        env_id,
        render_mode=chosen_render_mode,
        predefined_map_list=map_suite,
        activate_game_status=bool(env_config.get("activate_game_status", False)),
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_env_builder(config: dict[str, Any], render_mode: str | None = None):
    """Create a zero-argument environment factory closure.

    This shape is required by Stable-Baselines vectorized environment helpers
    and by evaluation callbacks that need fresh environments repeatedly.

    Args:
        config: Resolved experiment configuration dictionary.
        render_mode: Optional render mode override.

    Returns:
        Callable that constructs one configured environment.
    """
    def _builder() -> gym.Env:
        return make_env(config, render_mode=render_mode)

    return _builder
