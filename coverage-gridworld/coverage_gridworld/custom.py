from __future__ import annotations

import gymnasium as gym
import numpy as np

"""
Default observation/reward behavior used by the base environment.

The new experiment framework in `rl_coverage/` can still wrap the environment with
alternative observation and reward variants, but these defaults are now sensible
for quick smoke tests and manual play.
"""


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """Return a SB3-friendly flattened RGB observation space."""
    return gym.spaces.Box(
        low=0,
        high=255,
        shape=(env.grid_size * env.grid_size * 3,),
        dtype=np.uint8,
    )


def observation(grid: np.ndarray) -> np.ndarray:
    """Return the flattened RGB grid as a uint8 vector."""
    return grid.astype(np.uint8, copy=False).flatten()


def reward(info: dict) -> float:
    """
    A conservative dense default reward.

    This is intentionally simple because richer reward shaping is handled by the
    reusable reward wrappers in `rl_coverage.rewards`.
    """
    reward_value = -0.01

    if info["new_cell_covered"]:
        reward_value += 1.0
    else:
        reward_value -= 0.02

    if info["game_over"]:
        reward_value -= 5.0

    if info["cells_remaining"] == 0:
        reward_value += 10.0

    return float(reward_value)
