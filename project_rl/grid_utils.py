"""Grid parsing, masking, and geometry helpers for feature extraction.

The environment exposes grid observations as 10x10 RGB arrays. Observation-space
code needs to repeatedly answer questions such as:
- which cells are walls, enemies, or frontier cells?
- where is the agent currently located?
- how far is the nearest frontier in a given direction?
- which cells will be observed by enemies in future rotations?

This module centralizes those low-level operations so reward and observation code
can stay focused on feature design rather than pixel bookkeeping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

GRID_SIZE = 10

BLACK = np.array((0, 0, 0), dtype=np.uint8)
WHITE = np.array((255, 255, 255), dtype=np.uint8)
BROWN = np.array((101, 67, 33), dtype=np.uint8)
GREY = np.array((160, 161, 161), dtype=np.uint8)
GREEN = np.array((31, 198, 0), dtype=np.uint8)
RED = np.array((255, 0, 0), dtype=np.uint8)
LIGHT_RED = np.array((255, 127, 127), dtype=np.uint8)

DIRECTION_DELTAS = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0),
}

CATEGORY_IDS = {
    "unexplored": 0,
    "observed_unexplored": 1,
    "explored": 2,
    "observed_explored": 3,
    "wall": 4,
    "enemy": 5,
    "agent": 6,
    "out_of_bounds": 7,
}


@dataclass
class GridSummary:
    """Convenient boolean-mask summary of the current RGB grid state.

    Each field is a 10x10 boolean array describing one semantic slice of the
    environment. Observation builders use this dataclass instead of repeatedly
    re-parsing colors from the raw grid.
    """

    grid: np.ndarray
    unexplored: np.ndarray
    explored: np.ndarray
    walls: np.ndarray
    enemies: np.ndarray
    observed_unexplored: np.ndarray
    observed_explored: np.ndarray
    agent: np.ndarray
    observed_any: np.ndarray
    blocked: np.ndarray
    frontier: np.ndarray


def reshape_grid(observation: np.ndarray) -> np.ndarray:
    """Normalize an observation into the environment's canonical grid shape.

    Args:
        observation: Raw observation array, either already shaped as ``10x10x3``
            or flattened.

    Returns:
        ``uint8`` RGB grid with shape ``(10, 10, 3)``.
    """
    array = np.asarray(observation, dtype=np.uint8)
    if array.shape == (GRID_SIZE, GRID_SIZE, 3):
        return array
    return array.reshape(GRID_SIZE, GRID_SIZE, 3)


def mask_for_color(grid: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Return a boolean mask for all cells matching one RGB color exactly."""
    return np.all(grid == color, axis=-1)


def summarize_grid(grid: np.ndarray) -> GridSummary:
    """Convert a raw RGB grid into semantic masks used by feature builders.

    Args:
        grid: Canonical ``10x10x3`` environment grid.

    Returns:
        :class:`GridSummary` containing boolean masks for all important cell
        categories and combined helper masks such as ``blocked`` and ``frontier``.
    """
    unexplored = mask_for_color(grid, BLACK)
    explored = mask_for_color(grid, WHITE)
    walls = mask_for_color(grid, BROWN)
    enemies = mask_for_color(grid, GREEN)
    observed_unexplored = mask_for_color(grid, RED)
    observed_explored = mask_for_color(grid, LIGHT_RED)
    agent = mask_for_color(grid, GREY)
    observed_any = observed_unexplored | observed_explored
    blocked = walls | enemies
    frontier = unexplored | observed_unexplored

    return GridSummary(
        grid=grid,
        unexplored=unexplored,
        explored=explored,
        walls=walls,
        enemies=enemies,
        observed_unexplored=observed_unexplored,
        observed_explored=observed_explored,
        agent=agent,
        observed_any=observed_any,
        blocked=blocked,
        frontier=frontier,
    )


def in_bounds(row: int, col: int) -> bool:
    """Check whether a grid coordinate lies inside the 10x10 board."""
    return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE


def target_position(row: int, col: int, action: int) -> tuple[int, int]:
    """Compute the raw target cell for an action before collision handling.

    Action ``4`` is treated as a wait/no-op, so the target remains the current
    position.
    """
    if action == 4:
        return row, col
    delta_row, delta_col = DIRECTION_DELTAS[action]
    return row + delta_row, col + delta_col


def resolved_target_position(summary: GridSummary, origin: tuple[int, int], action: int) -> tuple[int, int]:
    """Resolve the actual cell reached after applying walls/enemy blocking.

    If the requested move would leave the board or enter a blocked cell, the
    agent effectively remains at the origin.
    """
    row, col = target_position(origin[0], origin[1], action)
    if not in_bounds(row, col) or summary.blocked[row, col]:
        return origin
    return row, col


def agent_position(summary: GridSummary, fallback_agent_pos: int | None = None) -> tuple[int, int]:
    """Locate the agent in row/column form.

    Args:
        summary: Parsed semantic grid summary.
        fallback_agent_pos: Optional flattened environment position used when the
            agent pixel is absent from the current observation.

    Returns:
        ``(row, col)`` tuple for the agent location.
    """
    position = np.argwhere(summary.agent)
    if position.size != 0:
        row, col = position[0]
        return int(row), int(col)
    if fallback_agent_pos is None:
        raise ValueError("Agent position not present in observation")
    return int(fallback_agent_pos) // GRID_SIZE, int(fallback_agent_pos) % GRID_SIZE


def category_id_at(summary: GridSummary, row: int, col: int) -> int:
    """Return a compact integer category id for one grid cell."""
    if row < 0 or col < 0 or row >= GRID_SIZE or col >= GRID_SIZE:
        return CATEGORY_IDS["out_of_bounds"]
    if summary.agent[row, col]:
        return CATEGORY_IDS["agent"]
    if summary.walls[row, col]:
        return CATEGORY_IDS["wall"]
    if summary.enemies[row, col]:
        return CATEGORY_IDS["enemy"]
    if summary.observed_unexplored[row, col]:
        return CATEGORY_IDS["observed_unexplored"]
    if summary.observed_explored[row, col]:
        return CATEGORY_IDS["observed_explored"]
    if summary.unexplored[row, col]:
        return CATEGORY_IDS["unexplored"]
    return CATEGORY_IDS["explored"]


def directional_free_run(summary: GridSummary, origin: tuple[int, int], action: int) -> int:
    """Measure how many free cells are available in one cardinal direction."""
    row, col = origin
    delta_row, delta_col = DIRECTION_DELTAS[action]
    distance = 0

    while True:
        row += delta_row
        col += delta_col
        if not in_bounds(row, col) or summary.blocked[row, col]:
            return distance
        distance += 1


def directional_target_distance(
    mask: np.ndarray,
    summary: GridSummary,
    origin: tuple[int, int],
    action: int,
) -> int | None:
    """Find the first distance to a target-mask cell in one direction.

    Returns ``None`` if the scan hits a wall/boundary before finding a target.
    """
    row, col = origin
    delta_row, delta_col = DIRECTION_DELTAS[action]
    distance = 0

    while True:
        row += delta_row
        col += delta_col
        distance += 1
        if not in_bounds(row, col) or summary.blocked[row, col]:
            return None
        if mask[row, col]:
            return distance


def directional_frontier_distance(summary: GridSummary, origin: tuple[int, int], action: int) -> int | None:
    """Specialized directional distance lookup for frontier cells."""
    return directional_target_distance(summary.frontier, summary, origin, action)


def nearest_position(mask: np.ndarray, origin: tuple[int, int]) -> tuple[tuple[int, int] | None, int | None]:
    """Find the Manhattan-nearest cell inside an arbitrary boolean mask."""
    positions = np.argwhere(mask)
    if positions.size == 0:
        return None, None

    best_position = None
    best_distance = None
    for row, col in positions:
        distance = abs(int(row) - origin[0]) + abs(int(col) - origin[1])
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_position = (int(row), int(col))

    return best_position, best_distance


def nearest_frontier(summary: GridSummary, origin: tuple[int, int]) -> tuple[tuple[int, int] | None, int | None]:
    """Find the nearest frontier cell using Manhattan distance."""
    return nearest_position(summary.frontier, origin)


def reachable_frontier_distance(summary: GridSummary, origin: tuple[int, int]) -> int | None:
    """Compute shortest-path distance to any reachable frontier using BFS."""
    if summary.frontier[origin]:
        return 0

    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    queue = deque([(origin[0], origin[1], 0)])
    visited[origin] = True

    while queue:
        row, col, distance = queue.popleft()
        for delta_row, delta_col in DIRECTION_DELTAS.values():
            next_row, next_col = row + delta_row, col + delta_col
            if not in_bounds(next_row, next_col):
                continue
            if visited[next_row, next_col] or summary.blocked[next_row, next_col]:
                continue
            if summary.frontier[next_row, next_col]:
                return distance + 1
            visited[next_row, next_col] = True
            queue.append((next_row, next_col, distance + 1))

    return None


def future_observation_masks(env: Any, horizon: int = 4) -> list[np.ndarray]:
    """Predict enemy field-of-view masks for the next few rotations.

    The environment rotates enemies deterministically each step. This helper rolls
    that rotation rule forward without changing environment state so temporal
    observation features can reason about near-future danger.
    """
    env_obj = env.unwrapped if hasattr(env, "unwrapped") else env
    grid = reshape_grid(np.asarray(env_obj.grid, dtype=np.uint8))
    summary = summarize_grid(grid)
    masks: list[np.ndarray] = []

    enemy_fov_distance = int(getattr(env_obj, "enemy_fov_distance", 4))

    for steps_ahead in range(1, horizon + 1):
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for enemy in getattr(env_obj, "enemy_list", []):
            orientation = (int(enemy.orientation) + steps_ahead) % 4
            for distance in range(1, enemy_fov_distance + 1):
                if orientation == 0:
                    row, col = enemy.y, enemy.x - distance
                elif orientation == 1:
                    row, col = enemy.y + distance, enemy.x
                elif orientation == 2:
                    row, col = enemy.y, enemy.x + distance
                else:
                    row, col = enemy.y - distance, enemy.x

                if not in_bounds(row, col) or summary.walls[row, col] or summary.enemies[row, col]:
                    break
                mask[row, col] = True
        masks.append(mask)

    return masks


def observation_countdown(masks: list[np.ndarray], position: tuple[int, int]) -> int | None:
    """Return how many future steps remain before a cell becomes observed."""
    row, col = position
    for index, mask in enumerate(masks, start=1):
        if mask[row, col]:
            return index
    return None


def safe_action_count(summary: GridSummary, future_mask: np.ndarray, origin: tuple[int, int]) -> int:
    """Count actions whose resolved target cell is safe under a future mask."""
    count = 0
    for action in range(5):
        row, col = resolved_target_position(summary, origin, action)
        if not future_mask[row, col]:
            count += 1
    return count
