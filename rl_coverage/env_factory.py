from __future__ import annotations

from typing import Any

from rl_coverage.bootstrap import *  # noqa: F401,F403
import gymnasium as gym
import coverage_gridworld  # noqa: F401

from rl_coverage.maps import resolve_map_suite
from rl_coverage.observations import wrap_observation
from rl_coverage.rewards import wrap_reward


def make_env(config: dict[str, Any], render_mode: str | None = None) -> gym.Env:
    env_config = config["environment"]
    map_suite = resolve_map_suite(env_config.get("map_suite"))
    env_id = env_config["id"] if map_suite is None else "standard"

    chosen_render_mode = render_mode if render_mode is not None else env_config.get("render_mode")
    if not chosen_render_mode:
        chosen_render_mode = None

    env = gym.make(
        env_id,
        render_mode=chosen_render_mode,
        predefined_map_list=map_suite,
        activate_game_status=env_config.get("activate_game_status", False),
    )
    env = wrap_reward(env, config["reward"])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = wrap_observation(env, config["observation"])
    return env


def make_env_builder(config: dict[str, Any], render_mode: str | None = None):
    def _builder() -> gym.Env:
        return make_env(config, render_mode=render_mode)

    return _builder
