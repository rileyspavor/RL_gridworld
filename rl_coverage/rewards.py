from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym

from rl_coverage.grid_utils import (
    future_observation_masks,
    observation_countdown,
    reshape_grid,
    resolved_target_position,
    safe_action_count,
    summarize_grid,
)


@dataclass
class RewardContext:
    action: int
    moved: bool
    invalid_move: bool
    stayed: bool
    revisited: bool
    recent_loop: bool
    new_cell_covered: bool
    coverage_delta: int
    total_covered_cells: int
    coverable_cells: int
    cells_remaining: int
    steps_remaining: int
    coverage_ratio: float
    current_cell_observed: bool
    game_over: bool
    success: bool
    timeout: bool
    unsafe_after_rotation: bool
    danger_countdown: int | None
    safe_actions_now: int
    safe_actions_next: int


RewardFunction = Callable[[RewardContext, dict[str, Any]], float]


def sparse_coverage_reward(context: RewardContext, params: dict[str, Any]) -> float:
    reward_value = float(params.get("step_penalty", -0.01))
    if context.new_cell_covered:
        reward_value += float(params.get("new_cell_reward", 1.0))
    if context.success:
        reward_value += float(params.get("success_bonus", 10.0))
    if context.game_over:
        reward_value += float(params.get("death_penalty", -5.0))
    if context.timeout and not context.success:
        reward_value += float(params.get("timeout_penalty", -1.0))
    return reward_value


def dense_coverage_reward(context: RewardContext, params: dict[str, Any]) -> float:
    reward_value = float(params.get("step_penalty", -0.02))
    reward_value += context.coverage_delta * float(params.get("new_cell_reward", 1.0))
    reward_value += context.coverage_delta * float(params.get("coverage_progress_weight", 2.5)) / max(1, context.coverable_cells)

    if context.invalid_move:
        reward_value += float(params.get("invalid_move_penalty", -0.15))
    if context.stayed:
        reward_value += float(params.get("stay_penalty", -0.08))
        forced_wait_threshold = int(params.get("forced_wait_safe_action_threshold", 1))
        if (not context.unsafe_after_rotation) and context.safe_actions_now <= forced_wait_threshold:
            reward_value += float(params.get("forced_wait_bonus", 0.0))
    if context.revisited:
        reward_value += float(params.get("revisit_penalty", -0.04))
    if context.recent_loop:
        reward_value += float(params.get("loop_penalty", -0.03))
    if context.current_cell_observed:
        reward_value += float(params.get("observed_cell_penalty", -0.15))
    if context.unsafe_after_rotation:
        reward_value += float(params.get("unsafe_after_rotation_penalty", 0.0))
    if context.danger_countdown is not None and context.danger_countdown > 1:
        reward_value += float(params.get("future_danger_penalty", 0.0)) / float(context.danger_countdown)
    if context.safe_actions_next <= int(params.get("trap_safe_action_threshold", 0)):
        reward_value += float(params.get("trap_penalty", 0.0))
    if context.success:
        reward_value += float(params.get("success_bonus", 15.0))
    if context.game_over:
        reward_value += float(params.get("death_penalty", -7.5))
    if context.timeout and not context.success:
        reward_value += float(params.get("timeout_penalty", -1.5))
    return reward_value


def survival_coverage_reward(context: RewardContext, params: dict[str, Any]) -> float:
    reward_value = dense_coverage_reward(context, {
        "step_penalty": params.get("step_penalty", -0.02),
        "new_cell_reward": params.get("new_cell_reward", 1.1),
        "coverage_progress_weight": params.get("coverage_progress_weight", 2.0),
        "invalid_move_penalty": params.get("invalid_move_penalty", -0.2),
        "stay_penalty": params.get("stay_penalty", -0.12),
        "revisit_penalty": params.get("revisit_penalty", -0.05),
        "loop_penalty": params.get("loop_penalty", -0.05),
        "observed_cell_penalty": params.get("observed_cell_penalty", -0.3),
        "unsafe_after_rotation_penalty": params.get("unsafe_after_rotation_penalty", -1.0),
        "future_danger_penalty": params.get("future_danger_penalty", -0.2),
        "forced_wait_bonus": params.get("forced_wait_bonus", 0.0),
        "forced_wait_safe_action_threshold": params.get("forced_wait_safe_action_threshold", 1),
        "trap_penalty": params.get("trap_penalty", -0.2),
        "trap_safe_action_threshold": params.get("trap_safe_action_threshold", 0),
        "success_bonus": params.get("success_bonus", 17.5),
        "death_penalty": params.get("death_penalty", -10.0),
        "timeout_penalty": params.get("timeout_penalty", -2.0),
    })
    reward_value += float(params.get("coverage_ratio_bonus", 0.0)) * context.coverage_ratio
    return reward_value


REWARD_VARIANTS: dict[str, RewardFunction] = {
    "sparse_coverage": sparse_coverage_reward,
    "dense_coverage": dense_coverage_reward,
    "survival_coverage": survival_coverage_reward,
}


class RewardVariantWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, variant_name: str, params: dict[str, Any] | None = None):
        super().__init__(env)
        if variant_name not in REWARD_VARIANTS:
            valid = ", ".join(sorted(REWARD_VARIANTS))
            raise KeyError(f"Unknown reward variant '{variant_name}'. Valid variants: {valid}")
        self.variant_name = variant_name
        self.params = dict(params or {})
        self.reward_fn = REWARD_VARIANTS[variant_name]
        self.position_visits: defaultdict[int, int] = defaultdict(int)
        self.recent_positions: deque[int] = deque(maxlen=int(self.params.get("loop_window", 8)))
        self.last_agent_pos = 0
        self.last_total_covered = 1

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        unwrapped = self.env.unwrapped
        self.position_visits = defaultdict(int)
        self.recent_positions = deque(maxlen=int(self.params.get("loop_window", 8)))
        self.last_agent_pos = int(unwrapped.agent_pos)
        self.last_total_covered = int(unwrapped.total_covered_cells)
        self.position_visits[self.last_agent_pos] += 1
        self.recent_positions.append(self.last_agent_pos)
        return observation, info

    def step(self, action: int):
        action = int(action)
        pre_env = self.env.unwrapped
        pre_summary = summarize_grid(reshape_grid(pre_env.grid))
        pre_origin = divmod(int(pre_env.agent_pos), pre_env.grid_size)
        future_masks = future_observation_masks(pre_env, horizon=4)
        target_pos = resolved_target_position(pre_summary, pre_origin, action)
        unsafe_after_rotation = bool(future_masks[0][target_pos]) if future_masks else False
        danger_countdown = observation_countdown(future_masks, target_pos) if future_masks else None
        safe_actions_now = safe_action_count(pre_summary, future_masks[0], pre_origin) if future_masks else 5
        branch_mask = future_masks[1] if len(future_masks) > 1 else (future_masks[0] if future_masks else None)
        safe_actions_next = safe_action_count(pre_summary, branch_mask, target_pos) if branch_mask is not None else 5

        observation, base_reward, terminated, truncated, info = self.env.step(action)
        current_pos = int(info["agent_pos"])
        total_covered = int(info["total_covered_cells"])
        coverage_delta = total_covered - self.last_total_covered
        moved = current_pos != self.last_agent_pos
        invalid_move = (action != 4) and not moved
        stayed = action == 4
        revisited = self.position_visits[current_pos] > 0 and not info["new_cell_covered"]

        recent_loop = current_pos in self.recent_positions
        self.position_visits[current_pos] += 1
        self.recent_positions.append(current_pos)
        self.last_agent_pos = current_pos
        self.last_total_covered = total_covered

        current_cell_observed = bool(info["game_over"])

        coverable = max(1, int(info["coverable_cells"]))
        success = info["cells_remaining"] == 0
        timeout = terminated and (not success) and (not info["game_over"]) and info["steps_remaining"] <= 0

        context = RewardContext(
            action=action,
            moved=moved,
            invalid_move=invalid_move,
            stayed=stayed,
            revisited=revisited,
            recent_loop=recent_loop,
            new_cell_covered=bool(info["new_cell_covered"]),
            coverage_delta=coverage_delta,
            total_covered_cells=total_covered,
            coverable_cells=coverable,
            cells_remaining=int(info["cells_remaining"]),
            steps_remaining=int(info["steps_remaining"]),
            coverage_ratio=total_covered / coverable,
            current_cell_observed=current_cell_observed,
            game_over=bool(info["game_over"]),
            success=success,
            timeout=timeout,
            unsafe_after_rotation=unsafe_after_rotation,
            danger_countdown=danger_countdown,
            safe_actions_now=safe_actions_now,
            safe_actions_next=safe_actions_next,
        )

        shaped_reward = self.reward_fn(context, self.params)
        shaped_reward += float(self.params.get("base_reward_scale", 0.0)) * float(base_reward)

        info = dict(info)
        info["base_reward"] = float(base_reward)
        info["shaped_reward"] = float(shaped_reward)
        info["reward_variant"] = self.variant_name
        info["coverage_ratio"] = context.coverage_ratio
        info["success"] = success
        info["timeout"] = timeout
        info["invalid_move"] = invalid_move
        info["revisited"] = revisited
        info["unsafe_after_rotation"] = unsafe_after_rotation
        info["danger_countdown"] = danger_countdown
        info["safe_actions_now"] = safe_actions_now
        info["safe_actions_next"] = safe_actions_next
        return observation, float(shaped_reward), terminated, truncated, info


def wrap_reward(env: gym.Env, config: dict[str, Any]) -> gym.Env:
    name = config["name"]
    if name == "native":
        return env
    return RewardVariantWrapper(env, variant_name=name, params=config.get("params", {}))
