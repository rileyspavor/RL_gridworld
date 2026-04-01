"""Reward-function registry for Coverage Gridworld experiments.

Reward variants consume normalized step context and return scalar rewards. The
module keeps reward definitions centralized so configs can switch reward shaping
without touching environment code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from project_rl.grid_utils import GRID_SIZE


@dataclass
class RewardInputs:
    """Normalized reward inputs derived from an environment step ``info`` payload."""

    total_covered_cells: int
    coverable_cells: int
    cells_remaining: int
    steps_remaining: int
    new_cell_covered: bool
    game_over: bool
    success: bool
    timeout: bool
    coverage_ratio: float
    observed_now: bool
    nearest_enemy_distance: int | None


RewardFunction = Callable[[RewardInputs, dict[str, Any]], float]


def _extract_inputs(info: dict[str, Any]) -> RewardInputs:
    """Convert raw environment ``info`` fields into :class:`RewardInputs`.

    Args:
        info: Environment info dictionary produced on each step.

    Returns:
        Structured reward inputs used by all reward variants.
    """
    coverable_cells = max(1, int(info["coverable_cells"]))
    total_covered_cells = int(info["total_covered_cells"])
    cells_remaining = int(info["cells_remaining"])
    steps_remaining = int(info["steps_remaining"])
    new_cell_covered = bool(info["new_cell_covered"])
    game_over = bool(info["game_over"])

    success = cells_remaining == 0
    timeout = (steps_remaining <= 0) and (not success) and (not game_over)
    coverage_ratio = total_covered_cells / coverable_cells

    agent_pos = int(info["agent_pos"])
    agent_row = agent_pos // GRID_SIZE
    agent_col = agent_pos % GRID_SIZE
    observed_now = False
    nearest_enemy_distance: int | None = None
    for enemy in info["enemies"]:
        enemy_distance = abs(int(enemy.y) - agent_row) + abs(int(enemy.x) - agent_col)
        if nearest_enemy_distance is None or enemy_distance < nearest_enemy_distance:
            nearest_enemy_distance = enemy_distance
        if (enemy.y, enemy.x) == (agent_row, agent_col):
            continue
        if (agent_row, agent_col) in enemy.get_fov_cells():
            observed_now = True

    return RewardInputs(
        total_covered_cells=total_covered_cells,
        coverable_cells=coverable_cells,
        cells_remaining=cells_remaining,
        steps_remaining=steps_remaining,
        new_cell_covered=new_cell_covered,
        game_over=game_over,
        success=success,
        timeout=timeout,
        coverage_ratio=coverage_ratio,
        observed_now=observed_now,
        nearest_enemy_distance=nearest_enemy_distance,
    )


def sparse_coverage(inputs: RewardInputs, params: dict[str, Any]) -> float:
    """Sparse reward that mainly values new coverage and terminal outcomes.

    Args:
        inputs: Normalized per-step reward inputs.
        params: Variant-specific coefficient overrides.

    Returns:
        Scalar reward for the current step.
    """
    reward_value = float(params.get("step_penalty", -0.01))

    if inputs.new_cell_covered:
        reward_value += float(params.get("new_cell_reward", 1.0))
    if inputs.success:
        reward_value += float(params.get("success_bonus", 10.0))
    if inputs.game_over:
        reward_value += float(params.get("death_penalty", -6.0))
    if inputs.timeout:
        reward_value += float(params.get("timeout_penalty", -1.0))

    return reward_value


def dense_coverage(inputs: RewardInputs, params: dict[str, Any]) -> float:
    """Dense reward balancing progress speed and final objective completion.

    Args:
        inputs: Normalized per-step reward inputs.
        params: Variant-specific coefficient overrides.

    Returns:
        Scalar reward for the current step.
    """
    reward_value = float(params.get("step_penalty", -0.02))
    reward_value += inputs.coverage_ratio * float(params.get("coverage_ratio_weight", 0.2))

    if inputs.new_cell_covered:
        reward_value += float(params.get("new_cell_reward", 1.2))
    else:
        reward_value += float(params.get("no_progress_penalty", -0.04))

    if inputs.success:
        reward_value += float(params.get("success_bonus", 14.0))
        reward_value += float(params.get("speed_bonus_weight", 2.0)) * (inputs.steps_remaining / 500.0)

    if inputs.game_over:
        reward_value += float(params.get("death_penalty", -8.0))
    if inputs.timeout:
        reward_value += float(params.get("timeout_penalty", -1.5))

    return reward_value


def survival_coverage(inputs: RewardInputs, params: dict[str, Any]) -> float:
    """Coverage reward with added penalties for enemy exposure/proximity.

    Args:
        inputs: Normalized per-step reward inputs.
        params: Variant-specific coefficient overrides.

    Returns:
        Scalar reward for the current step.
    """
    reward_value = float(params.get("step_penalty", -0.02))
    reward_value += inputs.coverage_ratio * float(params.get("coverage_ratio_weight", 0.15))

    if inputs.new_cell_covered:
        reward_value += float(params.get("new_cell_reward", 1.0))
    else:
        reward_value += float(params.get("no_progress_penalty", -0.05))

    if inputs.observed_now:
        reward_value += float(params.get("observed_cell_penalty", -0.25))

    if inputs.nearest_enemy_distance is not None and inputs.nearest_enemy_distance <= 2:
        pressure = (3 - inputs.nearest_enemy_distance) / 3.0
        reward_value += pressure * float(params.get("enemy_proximity_penalty", -0.2))

    if inputs.success:
        reward_value += float(params.get("success_bonus", 16.0))
        reward_value += float(params.get("speed_bonus_weight", 2.0)) * (inputs.steps_remaining / 500.0)

    if inputs.game_over:
        reward_value += float(params.get("death_penalty", -10.0))
    if inputs.timeout:
        reward_value += float(params.get("timeout_penalty", -2.0))

    return reward_value


REWARD_VARIANTS: dict[str, RewardFunction] = {
    "sparse_coverage": sparse_coverage,
    "dense_coverage": dense_coverage,
    "survival_coverage": survival_coverage,
}


def get_reward_variant(name: str) -> RewardFunction:
    """Lookup a reward variant implementation by registry name.

    Args:
        name: Key in :data:`REWARD_VARIANTS`.

    Returns:
        Callable reward function.

    Raises:
        KeyError: If the name is unknown.
    """
    if name not in REWARD_VARIANTS:
        valid = ", ".join(sorted(REWARD_VARIANTS))
        raise KeyError(f"Unknown reward variant '{name}'. Valid variants: {valid}")
    return REWARD_VARIANTS[name]


def compute_reward(name: str, info: dict[str, Any], params: dict[str, Any] | None = None) -> float:
    """Compute one step reward using a named reward variant.

    Args:
        name: Reward variant name.
        info: Environment info payload for the current step.
        params: Optional variant-parameter overrides.

    Returns:
        Scalar reward value.
    """
    reward_fn = get_reward_variant(name)
    inputs = _extract_inputs(info)
    return float(reward_fn(inputs, dict(params or {})))
