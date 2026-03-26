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
    0: (0, -1),   # left
    1: (1, 0),    # down
    2: (0, 1),    # right
    3: (-1, 0),   # up
}


@dataclass
class GridSummary:
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


def reshape_grid(observation: np.ndarray) -> np.ndarray:
    array = np.asarray(observation, dtype=np.uint8)
    return array.reshape(GRID_SIZE, GRID_SIZE, 3)


def mask_for_color(grid: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.all(grid == color, axis=-1)


def summarize_grid(grid: np.ndarray) -> GridSummary:
    unexplored = mask_for_color(grid, BLACK)
    explored = mask_for_color(grid, WHITE)
    walls = mask_for_color(grid, BROWN)
    enemies = mask_for_color(grid, GREEN)
    observed_unexplored = mask_for_color(grid, RED)
    observed_explored = mask_for_color(grid, LIGHT_RED)
    agent = mask_for_color(grid, GREY)
    observed_any = observed_unexplored | observed_explored
    blocked = walls | enemies

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
    )


def agent_position(summary: GridSummary) -> tuple[int, int]:
    location = np.argwhere(summary.agent)
    if location.size == 0:
        raise ValueError("Agent position not present in observation")
    row, col = location[0]
    return int(row), int(col)


def category_id_at(summary: GridSummary, row: int, col: int) -> int:
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


def target_position(row: int, col: int, action: int) -> tuple[int, int]:
    if action == 4:
        return row, col
    dy, dx = DIRECTION_DELTAS[action]
    return row + dy, col + dx


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_position(mask: np.ndarray, origin: tuple[int, int]) -> tuple[tuple[int, int] | None, int | None]:
    positions = np.argwhere(mask)
    if positions.size == 0:
        return None, None
    best_pos = None
    best_distance = None
    for row, col in positions:
        dist = manhattan_distance(origin, (int(row), int(col)))
        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_pos = (int(row), int(col))
    return best_pos, best_distance


def directional_free_run(summary: GridSummary, origin: tuple[int, int], action: int) -> int:
    row, col = origin
    dy, dx = DIRECTION_DELTAS[action]
    distance = 0
    while True:
        row += dy
        col += dx
        if not in_bounds(row, col) or summary.blocked[row, col]:
            return distance
        distance += 1


def directional_target_distance(mask: np.ndarray, summary: GridSummary, origin: tuple[int, int], action: int) -> int | None:
    row, col = origin
    dy, dx = DIRECTION_DELTAS[action]
    distance = 0
    while True:
        row += dy
        col += dx
        distance += 1
        if not in_bounds(row, col) or summary.blocked[row, col]:
            return None
        if mask[row, col]:
            return distance


def reachable_frontier_distance(summary: GridSummary, origin: tuple[int, int]) -> int | None:
    frontier = summary.unexplored | summary.observed_unexplored
    if frontier[origin]:
        return 0

    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    queue = deque([(origin[0], origin[1], 0)])
    visited[origin] = True

    while queue:
        row, col, distance = queue.popleft()
        for dy, dx in DIRECTION_DELTAS.values():
            next_row, next_col = row + dy, col + dx
            if not in_bounds(next_row, next_col):
                continue
            if visited[next_row, next_col] or summary.blocked[next_row, next_col]:
                continue
            if frontier[next_row, next_col]:
                return distance + 1
            visited[next_row, next_col] = True
            queue.append((next_row, next_col, distance + 1))

    return None


def resolved_target_position(summary: GridSummary, origin: tuple[int, int], action: int) -> tuple[int, int]:
    if action == 4:
        return origin
    row, col = target_position(origin[0], origin[1], action)
    if not in_bounds(row, col) or summary.blocked[row, col]:
        return origin
    return row, col


def future_observation_masks(env: Any, horizon: int = 4) -> list[np.ndarray]:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    grid = np.asarray(unwrapped.grid, dtype=np.uint8)
    summary = summarize_grid(grid)

    masks: list[np.ndarray] = []
    for steps_ahead in range(1, horizon + 1):
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for enemy in unwrapped.enemy_list:
            orientation = (int(enemy.orientation) + steps_ahead) % 4
            for distance in range(1, int(unwrapped.enemy_fov_distance) + 1):
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
    row, col = position
    for index, mask in enumerate(masks, start=1):
        if mask[row, col]:
            return index
    return None


def safe_action_count(summary: GridSummary, future_mask: np.ndarray, origin: tuple[int, int]) -> int:
    count = 0
    for action in range(5):
        row, col = resolved_target_position(summary, origin, action)
        if not future_mask[row, col]:
            count += 1
    return count
