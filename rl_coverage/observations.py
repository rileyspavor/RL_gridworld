from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from rl_coverage.grid_utils import (
    CATEGORY_IDS,
    GRID_SIZE,
    agent_position,
    category_id_at,
    directional_free_run,
    directional_target_distance,
    future_observation_masks,
    in_bounds,
    nearest_position,
    observation_countdown,
    reachable_frontier_distance,
    reshape_grid,
    resolved_target_position,
    safe_action_count,
    summarize_grid,
    target_position,
)
from rl_coverage.maps import STANDARD_MAP_ORDER, STANDARD_MAPS

MAX_ENEMY_FEATURE_SLOTS = 5

_STANDARD_MAP_SIGNATURES = {
    tuple(tuple(int(cell) for cell in row) for row in layout): name
    for name, layout in STANDARD_MAPS.items()
}


def _agent_position(summary, env: gym.Env) -> tuple[int, int]:
    try:
        return agent_position(summary)
    except ValueError:
        agent_pos = int(env.unwrapped.agent_pos)
        return agent_pos // GRID_SIZE, agent_pos % GRID_SIZE


def _map_name_for_env(env: gym.Env) -> str | None:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    current_map = getattr(unwrapped, "predefined_map", None)

    current_index = getattr(unwrapped, "current_predefined_map", None)
    predefined_map_list = getattr(unwrapped, "predefined_map_list", None)
    if isinstance(current_index, int) and isinstance(predefined_map_list, list):
        zero_based_index = current_index - 1
        if 0 <= zero_based_index < len(predefined_map_list):
            current_map = predefined_map_list[zero_based_index]

    if current_map is None:
        return None
    signature = tuple(tuple(int(cell) for cell in row) for row in current_map)
    return _STANDARD_MAP_SIGNATURES.get(signature)


def _map_identity_features(env: gym.Env) -> list[float]:
    map_name = _map_name_for_env(env)
    return [1.0 if name == map_name else 0.0 for name in STANDARD_MAP_ORDER]


def _reachable_action_metrics(
    summary,
    origin: tuple[int, int],
    frontier_mask: np.ndarray,
) -> list[float]:
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


@dataclass
class ObservationTransform:
    name: str
    params: dict[str, Any]

    def observation_space(self, env: gym.Env) -> gym.spaces.Space:
        raise NotImplementedError

    def transform(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        raise NotImplementedError


class RawRgbTransform(ObservationTransform):
    def observation_space(self, env: gym.Env) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(GRID_SIZE * GRID_SIZE * 3,),
            dtype=np.float32,
        )

    def transform(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32) / 255.0


class LayeredGridTransform(ObservationTransform):
    def observation_space(self, env: gym.Env) -> gym.spaces.Space:
        feature_count = 7 * GRID_SIZE * GRID_SIZE + 5
        return gym.spaces.Box(low=0.0, high=1.0, shape=(feature_count,), dtype=np.float32)

    def transform(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        grid = reshape_grid(observation)
        summary = summarize_grid(grid)
        row, col = _agent_position(summary, env)

        channels = [
            summary.unexplored,
            summary.observed_unexplored,
            summary.explored,
            summary.observed_explored,
            summary.walls,
            summary.enemies,
            summary.agent,
        ]
        flattened_channels = [channel.astype(np.float32).reshape(-1) for channel in channels]

        coverable = max(1, int(env.unwrapped.coverable_cells))
        scalars = np.array(
            [
                row / (GRID_SIZE - 1),
                col / (GRID_SIZE - 1),
                float(env.unwrapped.total_covered_cells) / coverable,
                float(env.unwrapped.coverable_cells - env.unwrapped.total_covered_cells) / coverable,
                float(env.unwrapped.steps_remaining) / 500.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([*flattened_channels, scalars]).astype(np.float32, copy=False)


def _frontier_core_features(summary, env: gym.Env, origin: tuple[int, int]) -> list[float]:
    row, col = origin
    coverable = max(1, int(env.unwrapped.coverable_cells))

    features: list[float] = [
        row / (GRID_SIZE - 1),
        col / (GRID_SIZE - 1),
        float(env.unwrapped.total_covered_cells) / coverable,
        float(env.unwrapped.coverable_cells - env.unwrapped.total_covered_cells) / coverable,
        float(env.unwrapped.steps_remaining) / 500.0,
    ]

    for action in range(4):
        target_row, target_col = target_position(row, col, action)
        in_bounds_target = 0 <= target_row < GRID_SIZE and 0 <= target_col < GRID_SIZE
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
    if nearest_frontier_pos is None:
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
        quadrant_frontier = frontier_mask[row_slice, col_slice]
        features.append(float(quadrant_frontier.mean()))

    reachable_distance = reachable_frontier_distance(summary, origin)
    features.append(-1.0 if reachable_distance is None else reachable_distance / (2 * (GRID_SIZE - 1)))
    features.append(float(summary.observed_any[row, col]))
    return features


class FrontierFeaturesTransform(ObservationTransform):
    FEATURE_COUNT = 54

    def observation_space(self, env: gym.Env) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.FEATURE_COUNT,), dtype=np.float32)

    def transform(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        grid = reshape_grid(observation)
        summary = summarize_grid(grid)
        origin = _agent_position(summary, env)
        features = _frontier_core_features(summary, env, origin)
        return np.asarray(features, dtype=np.float32)


class TemporalFrontierFeaturesTransform(ObservationTransform):
    FEATURE_COUNT = 122

    def observation_space(self, env: gym.Env) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.FEATURE_COUNT,), dtype=np.float32)

    def transform(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        grid = reshape_grid(observation)
        summary = summarize_grid(grid)
        origin = _agent_position(summary, env)
        features = _frontier_core_features(summary, env, origin)

        frontier_mask = summary.unexplored | summary.observed_unexplored
        future_masks = future_observation_masks(env, horizon=4)
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

        enemy_count = max(1, len(env.unwrapped.enemy_list))
        for orientation in range(4):
            matches = sum(int(enemy.orientation == orientation) for enemy in env.unwrapped.enemy_list)
            features.append(matches / enemy_count)

        row, col = origin
        enemies = list(env.unwrapped.enemy_list)[:MAX_ENEMY_FEATURE_SLOTS]
        for enemy in enemies:
            features.extend(
                [
                    (enemy.y - row) / (GRID_SIZE - 1),
                    (enemy.x - col) / (GRID_SIZE - 1),
                    float(enemy.orientation == 0),
                    float(enemy.orientation == 1),
                    float(enemy.orientation == 2),
                    float(enemy.orientation == 3),
                ]
            )
        for _ in range(MAX_ENEMY_FEATURE_SLOTS - len(enemies)):
            features.extend([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0])

        return np.asarray(features, dtype=np.float32)


class StrategicTemporalFrontierFeaturesTransform(ObservationTransform):
    FEATURE_COUNT = 142

    def observation_space(self, env: gym.Env) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.FEATURE_COUNT,), dtype=np.float32)

    def transform(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        grid = reshape_grid(observation)
        summary = summarize_grid(grid)
        origin = _agent_position(summary, env)
        frontier_mask = summary.unexplored | summary.observed_unexplored

        features = list(TemporalFrontierFeaturesTransform(name=self.name, params=self.params).transform(observation, env))
        features.extend(_reachable_action_metrics(summary, origin, frontier_mask))
        features.extend(_map_identity_features(env))
        return np.asarray(features, dtype=np.float32)


TRANSFORMS = {
    "raw_rgb": RawRgbTransform,
    "layered_grid": LayeredGridTransform,
    "frontier_features": FrontierFeaturesTransform,
    "temporal_frontier_features": TemporalFrontierFeaturesTransform,
    "strategic_temporal_frontier_features": StrategicTemporalFrontierFeaturesTransform,
}


class ObservationVariantWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, transform: ObservationTransform):
        super().__init__(env)
        self.transformer = transform
        self.observation_space = transform.observation_space(env)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return self.transformer.transform(observation, self.env)


def build_transform(config: dict[str, Any]) -> ObservationTransform:
    name = config["name"]
    params = dict(config.get("params", {}))
    if name == "native":
        return RawRgbTransform(name="raw_rgb", params=params)
    if name not in TRANSFORMS:
        valid = ", ".join(sorted(TRANSFORMS))
        raise KeyError(f"Unknown observation variant '{name}'. Valid variants: {valid}")
    return TRANSFORMS[name](name=name, params=params)


def wrap_observation(env: gym.Env, config: dict[str, Any]) -> gym.Env:
    name = config["name"]
    if name == "native":
        return env
    return ObservationVariantWrapper(env, build_transform(config))
