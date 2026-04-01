"""Observation-space implementations used throughout the project.

The assignment experiments use two main feature families:
- ``frontier_features`` for compact exploration-oriented state summaries
- ``temporal_frontier_features`` for exploration plus short-horizon danger cues

This module also contains ``strategic_temporal_frontier_features``, a compatibility
observation used only to replay the recovered ``comp_model_80`` checkpoint. That
path is intentionally separated from the clean 2x3 assignment matrix.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from project_rl.grid_utils import (
    CATEGORY_IDS,
    GRID_SIZE,
    agent_position,
    category_id_at,
    directional_free_run,
    directional_frontier_distance,
    directional_target_distance,
    future_observation_masks,
    in_bounds,
    nearest_frontier,
    nearest_position,
    observation_countdown,
    reachable_frontier_distance,
    reshape_grid,
    resolved_target_position,
    safe_action_count,
    summarize_grid,
    target_position,
)
from project_rl.maps import STANDARD_MAP_ORDER, STANDARD_MAPS

ObservationSpaceFn = Callable[[gym.Env, dict[str, Any]], gym.spaces.Space]
ObservationTransformFn = Callable[[np.ndarray, Any | None, dict[str, Any]], np.ndarray]
MAX_ENEMY_FEATURE_SLOTS = 5

_STANDARD_MAP_SIGNATURES = {
    tuple(tuple(int(cell) for cell in row) for row in layout): name
    for name, layout in STANDARD_MAPS.items()
}


def _normalized_count(mask: np.ndarray) -> float:
    """Convert a boolean mask into a fraction of total grid cells."""
    return float(mask.sum()) / float(GRID_SIZE * GRID_SIZE)


def raw_grid_observation_space(env: gym.Env, params: dict[str, Any]) -> gym.spaces.Space:
    """Observation space for the flattened raw RGB grid baseline."""
    return gym.spaces.Box(
        low=0.0,
        high=255.0,
        shape=(GRID_SIZE * GRID_SIZE * 3,),
        dtype=np.float32,
    )


def raw_grid_observation(grid: np.ndarray, env: Any | None, params: dict[str, Any]) -> np.ndarray:
    """Flatten the full RGB grid into a vector baseline observation.

    This variant is mainly useful as a simple reference point against the more
    engineered feature observations.
    """
    array = reshape_grid(grid).reshape(-1).astype(np.float32)
    normalize = bool(params.get("normalize", False))
    if normalize:
        return array / 255.0
    return array


def _cell_score(summary, row: int, col: int) -> float:
    """Map local cell semantics to a small signed score for neighborhood features."""
    if not in_bounds(row, col):
        return -1.0
    if summary.agent[row, col]:
        return 1.0
    if summary.walls[row, col] or summary.enemies[row, col]:
        return -0.75
    if summary.unexplored[row, col] or summary.observed_unexplored[row, col]:
        return 0.75
    if summary.observed_explored[row, col]:
        return -0.25
    return 0.25


def _frontier_core_features(grid: np.ndarray, env: Any | None) -> list[float]:
    """Build the 40-feature exploration-focused representation.

    This shared core captures the agent position, global coverage progress, local
    occupancy around the agent, directional movement affordances, and nearest
    frontier cues. It is the base representation for both assignment observation
    spaces.
    """
    summary = summarize_grid(reshape_grid(grid))
    fallback_pos = int(getattr(env, "agent_pos", 0)) if env is not None else 0
    row, col = agent_position(summary, fallback_agent_pos=fallback_pos)

    covered_cells = int((summary.explored | summary.observed_explored | summary.agent).sum())
    coverable_cells = int(getattr(env, "coverable_cells", covered_cells + int(summary.frontier.sum())))
    coverable_cells = max(1, coverable_cells)
    coverage_ratio = covered_cells / coverable_cells

    features: list[float] = [
        row / (GRID_SIZE - 1),
        col / (GRID_SIZE - 1),
        coverage_ratio,
        1.0 - coverage_ratio,
        _normalized_count(summary.frontier),
        _normalized_count(summary.observed_any),
        _normalized_count(summary.walls),
        _normalized_count(summary.enemies),
        _normalized_count(summary.explored | summary.observed_explored),
    ]

    origin = (row, col)
    for action in range(4):
        target_row, target_col = target_position(row, col, action)
        blocked = (not in_bounds(target_row, target_col)) or summary.blocked[target_row, target_col]
        immediate_frontier = 0.0
        immediate_observed = 0.0
        if not blocked:
            immediate_frontier = float(summary.frontier[target_row, target_col])
            immediate_observed = float(summary.observed_any[target_row, target_col])

        free_run = directional_free_run(summary, origin, action) / float(GRID_SIZE - 1)
        frontier_distance = directional_frontier_distance(summary, origin, action)
        normalized_frontier_distance = -1.0
        if frontier_distance is not None:
            normalized_frontier_distance = frontier_distance / float(GRID_SIZE - 1)

        features.extend(
            [
                0.0 if blocked else 1.0,
                immediate_frontier,
                immediate_observed,
                free_run,
                normalized_frontier_distance,
            ]
        )

    for local_row in range(row - 1, row + 2):
        for local_col in range(col - 1, col + 2):
            if local_row == row and local_col == col:
                continue
            features.append(_cell_score(summary, local_row, local_col))

    nearest_pos, nearest_distance = nearest_frontier(summary, origin)
    if nearest_pos is None or nearest_distance is None:
        features.extend([-1.0, -1.0, -1.0])
    else:
        features.extend(
            [
                (nearest_pos[0] - row) / float(GRID_SIZE - 1),
                (nearest_pos[1] - col) / float(GRID_SIZE - 1),
                nearest_distance / float(2 * (GRID_SIZE - 1)),
            ]
        )

    return features


def frontier_features_observation_space(env: gym.Env, params: dict[str, Any]) -> gym.spaces.Space:
    """Space definition for the compact 40-dimensional frontier observation."""
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(40,), dtype=np.float32)


def frontier_features_observation(grid: np.ndarray, env: Any | None, params: dict[str, Any]) -> np.ndarray:
    """Return the compact frontier-feature vector used in the clean matrix."""
    return np.asarray(_frontier_core_features(grid, env), dtype=np.float32)


def temporal_frontier_feature_count(params: dict[str, Any]) -> int:
    """Compute feature length for the temporal frontier variant from its params."""
    horizon = int(params.get("horizon", 4))
    max_enemy_slots = int(params.get("max_enemy_slots", 5))
    return 40 + (horizon * 5) + horizon + 4 + (max_enemy_slots * 6)


def temporal_frontier_observation_space(env: gym.Env, params: dict[str, Any]) -> gym.spaces.Space:
    """Space definition for the temporal frontier feature vector."""
    return gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(temporal_frontier_feature_count(params),),
        dtype=np.float32,
    )


def temporal_frontier_observation(grid: np.ndarray, env: Any | None, params: dict[str, Any]) -> np.ndarray:
    """Extend frontier features with short-horizon enemy-visibility forecasts.

    The added features describe how dangerous each candidate action becomes over a
    few future enemy rotations, along with compact enemy orientation/location
    summaries. This is the richer assignment observation that performed best.
    """
    horizon = int(params.get("horizon", 4))
    max_enemy_slots = int(params.get("max_enemy_slots", 5))

    summary = summarize_grid(reshape_grid(grid))
    fallback_pos = int(getattr(env, "agent_pos", 0)) if env is not None else 0
    origin = agent_position(summary, fallback_agent_pos=fallback_pos)

    features = _frontier_core_features(grid, env)
    target_positions = [resolved_target_position(summary, origin, action) for action in range(5)]

    if env is None:
        future_masks = [np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool) for _ in range(horizon)]
        enemies = []
    else:
        future_masks = future_observation_masks(env, horizon=horizon)
        env_obj = env.unwrapped if hasattr(env, "unwrapped") else env
        enemies = list(getattr(env_obj, "enemy_list", []))

    for mask in future_masks:
        for row, col in target_positions:
            features.append(float(mask[row, col]))

    for mask in future_masks:
        safe_actions = 0
        for row, col in target_positions:
            if not mask[row, col]:
                safe_actions += 1
        features.append(safe_actions / 5.0)

    enemy_count = max(1, len(enemies))
    for orientation in range(4):
        orientation_matches = sum(int(int(enemy.orientation) == orientation) for enemy in enemies)
        features.append(orientation_matches / enemy_count)

    row, col = origin
    for enemy in enemies[:max_enemy_slots]:
        features.extend(
            [
                (enemy.y - row) / float(GRID_SIZE - 1),
                (enemy.x - col) / float(GRID_SIZE - 1),
                float(int(enemy.orientation) == 0),
                float(int(enemy.orientation) == 1),
                float(int(enemy.orientation) == 2),
                float(int(enemy.orientation) == 3),
            ]
        )

    for _ in range(max_enemy_slots - len(enemies)):
        features.extend([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0])

    return np.asarray(features, dtype=np.float32)


def _compat_frontier_core_features(grid: np.ndarray, env: Any | None) -> list[float]:
    """Rebuild the legacy core features expected by ``comp_model_80``.

    This path is intentionally separate from the clean assignment observations.
    Its job is to recover the richer handcrafted features that the archived model
    was originally trained on.
    """
    summary = summarize_grid(reshape_grid(grid))
    env_obj = env.unwrapped if hasattr(env, "unwrapped") else env
    fallback_pos = int(getattr(env_obj, "agent_pos", 0)) if env_obj is not None else 0
    row, col = agent_position(summary, fallback_agent_pos=fallback_pos)

    coverable = max(1, int(getattr(env_obj, "coverable_cells", max(1, int((~summary.blocked).sum())))))
    total_covered = int(getattr(env_obj, "total_covered_cells", int((summary.explored | summary.observed_explored | summary.agent).sum())))
    steps_remaining = float(getattr(env_obj, "steps_remaining", 500.0)) / 500.0

    features: list[float] = [
        row / (GRID_SIZE - 1),
        col / (GRID_SIZE - 1),
        float(total_covered) / coverable,
        float(max(0, coverable - total_covered)) / coverable,
        steps_remaining,
    ]

    origin = (row, col)
    for action in range(4):
        target_row, target_col = target_position(row, col, action)
        in_bounds_target = in_bounds(target_row, target_col)
        blocked = int((not in_bounds_target) or summary.blocked[target_row, target_col])
        category = category_id_at(summary, target_row, target_col)
        features.extend(
            [
                1.0 - blocked,
                float(blocked),
                float(category == CATEGORY_IDS["unexplored"]),
                float(category == CATEGORY_IDS["observed_unexplored"]),
                float(category in {CATEGORY_IDS["explored"], CATEGORY_IDS["observed_explored"], CATEGORY_IDS["agent"]}),
            ]
        )

    for local_row in range(row - 1, row + 2):
        for local_col in range(col - 1, col + 2):
            if local_row == row and local_col == col:
                continue
            category = category_id_at(summary, local_row, local_col)
            features.append((category / 7.0) * 2.0 - 1.0)

    frontier_mask = summary.unexplored | summary.observed_unexplored
    for action in range(4):
        free_run = directional_free_run(summary, origin, action) / (GRID_SIZE - 1)
        frontier_distance = directional_target_distance(frontier_mask, summary, origin, action)
        observed_distance = directional_target_distance(summary.observed_any, summary, origin, action)
        features.extend(
            [
                free_run,
                -1.0 if frontier_distance is None else frontier_distance / (GRID_SIZE - 1),
                -1.0 if observed_distance is None else observed_distance / (GRID_SIZE - 1),
            ]
        )

    nearest_frontier_pos, nearest_frontier_distance = nearest_position(frontier_mask, origin)
    if nearest_frontier_pos is None or nearest_frontier_distance is None:
        features.extend([-1.0, -1.0, -1.0])
    else:
        features.extend(
            [
                (nearest_frontier_pos[0] - row) / (GRID_SIZE - 1),
                (nearest_frontier_pos[1] - col) / (GRID_SIZE - 1),
                nearest_frontier_distance / (2 * (GRID_SIZE - 1)),
            ]
        )

    for row_slice, col_slice in (
        (slice(0, 5), slice(0, 5)),
        (slice(0, 5), slice(5, 10)),
        (slice(5, 10), slice(0, 5)),
        (slice(5, 10), slice(5, 10)),
    ):
        features.append(float(frontier_mask[row_slice, col_slice].mean()))

    reachable_distance = reachable_frontier_distance(summary, origin)
    features.append(-1.0 if reachable_distance is None else reachable_distance / (2 * (GRID_SIZE - 1)))
    features.append(float(summary.observed_any[row, col]))
    return features


def _map_name_for_env(env: Any | None) -> str | None:
    """Infer the current standard-map name from environment map data."""
    if env is None:
        return None
    env_obj = env.unwrapped if hasattr(env, "unwrapped") else env
    current_map = getattr(env_obj, "predefined_map", None)

    current_index = getattr(env_obj, "current_predefined_map", None)
    predefined_map_list = getattr(env_obj, "predefined_map_list", None)
    if isinstance(current_index, int) and isinstance(predefined_map_list, list):
        zero_based_index = current_index - 1
        if 0 <= zero_based_index < len(predefined_map_list):
            current_map = predefined_map_list[zero_based_index]

    if current_map is None:
        return None
    signature = tuple(tuple(int(cell) for cell in row) for row in current_map)
    return _STANDARD_MAP_SIGNATURES.get(signature)


def _map_identity_features(env: Any | None) -> list[float]:
    """Return a one-hot encoding of the current standard map identity."""
    map_name = _map_name_for_env(env)
    return [1.0 if name == map_name else 0.0 for name in STANDARD_MAP_ORDER]


def _reachable_action_metrics(summary, origin: tuple[int, int], frontier_mask: np.ndarray) -> list[float]:
    """Describe how much traversable/frontier area each action can reach.

    These metrics are only used by the recovered compatibility observation and help
    the policy reason about longer-range consequences of local movement choices.
    """
    total_frontier = max(1, int(frontier_mask.sum()))
    total_traversable = max(1, int((~summary.blocked).sum()))
    metrics: list[float] = []

    for action in range(5):
        start = resolved_target_position(summary, origin, action)
        visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        queue = deque([(start[0], start[1], 0)])
        visited[start] = True
        component_cells = 0
        reachable_frontier = 0
        nearest_frontier_distance = None

        while queue:
            row, col, distance = queue.popleft()
            component_cells += 1
            if frontier_mask[row, col]:
                reachable_frontier += 1
                if nearest_frontier_distance is None:
                    nearest_frontier_distance = distance

            for next_action in range(4):
                next_row, next_col = target_position(row, col, next_action)
                if not in_bounds(next_row, next_col):
                    continue
                if visited[next_row, next_col] or summary.blocked[next_row, next_col]:
                    continue
                visited[next_row, next_col] = True
                queue.append((next_row, next_col, distance + 1))

        metrics.extend(
            [
                component_cells / total_traversable,
                reachable_frontier / total_frontier,
                -1.0 if nearest_frontier_distance is None else nearest_frontier_distance / (2 * (GRID_SIZE - 1)),
            ]
        )

    return metrics


def strategic_temporal_frontier_features_observation_space(env: gym.Env, params: dict[str, Any]) -> gym.spaces.Space:
    """Space definition for the 142-dimensional recovered-model observation."""
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(142,), dtype=np.float32)


def strategic_temporal_frontier_features_observation(grid: np.ndarray, env: Any | None, params: dict[str, Any]) -> np.ndarray:
    """Reconstruct the richer handcrafted observation used by ``comp_model_80``.

    Besides temporal safety features, this variant adds reachable-component
    statistics and map-identity information. It is kept for compatibility rather
    than as part of the clean assignment matrix.
    """
    summary = summarize_grid(reshape_grid(grid))
    env_obj = env.unwrapped if hasattr(env, "unwrapped") else env
    fallback_pos = int(getattr(env_obj, "agent_pos", 0)) if env_obj is not None else 0
    origin = agent_position(summary, fallback_agent_pos=fallback_pos)
    frontier_mask = summary.unexplored | summary.observed_unexplored

    features = _compat_frontier_core_features(grid, env)

    if env_obj is None:
        future_masks = [np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool) for _ in range(4)]
        enemies = []
    else:
        future_masks = future_observation_masks(env_obj, horizon=4)
        enemies = list(getattr(env_obj, "enemy_list", []))

    branch_mask = future_masks[1] if len(future_masks) > 1 else future_masks[-1]

    for action in range(5):
        target = resolved_target_position(summary, origin, action)
        for mask in future_masks:
            features.append(1.0 if not mask[target] else 0.0)
        countdown = observation_countdown(future_masks, target)
        if countdown is None:
            features.append(1.0)
        else:
            features.append((countdown - 1) / max(1, len(future_masks) - 1))

    for action in range(5):
        target = resolved_target_position(summary, origin, action)
        features.append(safe_action_count(summary, branch_mask, target) / 5.0)

    frontier_total = int(frontier_mask.sum())
    for mask in future_masks:
        if frontier_total == 0:
            features.append(1.0)
        else:
            features.append(float((frontier_mask & ~mask).sum()) / frontier_total)

    enemy_count = max(1, len(enemies))
    for orientation in range(4):
        matches = sum(int(int(enemy.orientation) == orientation) for enemy in enemies)
        features.append(matches / enemy_count)

    row, col = origin
    for enemy in enemies[:MAX_ENEMY_FEATURE_SLOTS]:
        features.extend(
            [
                (enemy.y - row) / (GRID_SIZE - 1),
                (enemy.x - col) / (GRID_SIZE - 1),
                float(int(enemy.orientation) == 0),
                float(int(enemy.orientation) == 1),
                float(int(enemy.orientation) == 2),
                float(int(enemy.orientation) == 3),
            ]
        )
    for _ in range(MAX_ENEMY_FEATURE_SLOTS - len(enemies)):
        features.extend([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0])

    features.extend(_reachable_action_metrics(summary, origin, frontier_mask))
    features.extend(_map_identity_features(env))
    return np.asarray(features, dtype=np.float32)


OBSERVATION_VARIANTS: dict[str, tuple[ObservationSpaceFn, ObservationTransformFn]] = {
    "raw_grid": (raw_grid_observation_space, raw_grid_observation),
    "frontier_features": (frontier_features_observation_space, frontier_features_observation),
    "temporal_frontier_features": (temporal_frontier_observation_space, temporal_frontier_observation),
    "strategic_temporal_frontier_features": (
        strategic_temporal_frontier_features_observation_space,
        strategic_temporal_frontier_features_observation,
    ),
}


def get_observation_variant(name: str) -> tuple[ObservationSpaceFn, ObservationTransformFn]:
    """Lookup an observation-space implementation by registry name.

    Args:
        name: Observation variant key from ``OBSERVATION_VARIANTS``.

    Returns:
        Pair of callables: one for the Gymnasium space and one for the transform.
    """
    if name not in OBSERVATION_VARIANTS:
        valid = ", ".join(sorted(OBSERVATION_VARIANTS))
        raise KeyError(f"Unknown observation variant '{name}'. Valid variants: {valid}")
    return OBSERVATION_VARIANTS[name]
