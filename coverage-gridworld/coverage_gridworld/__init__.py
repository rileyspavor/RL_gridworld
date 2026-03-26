from __future__ import annotations

from copy import deepcopy
from gymnasium.envs.registration import register
from coverage_gridworld.env import CoverageGridworld

JUST_GO_MAP = [
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

SAFE_MAP = [
    [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
]

MAZE_MAP = [
    [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
]

CHOKEPOINT_MAP = [
    [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
]

SNEAKY_ENEMIES_MAP = [
    [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
]

STANDARD_MAPS = {
    "just_go": JUST_GO_MAP,
    "safe": SAFE_MAP,
    "maze": MAZE_MAP,
    "chokepoint": CHOKEPOINT_MAP,
    "sneaky_enemies": SNEAKY_ENEMIES_MAP,
}

STANDARD_MAP_ORDER = tuple(STANDARD_MAPS.keys())


def clone_map(map_name: str) -> list[list[int]]:
    return deepcopy(STANDARD_MAPS[map_name])


def clone_map_suite(map_names: list[str] | tuple[str, ...]) -> list[list[list[int]]]:
    return [clone_map(name) for name in map_names]


register(
    id="standard",
    entry_point="coverage_gridworld:CoverageGridworld",
)

for env_id, predefined_map in STANDARD_MAPS.items():
    register(
        id=env_id,
        entry_point="coverage_gridworld:CoverageGridworld",
        kwargs={"predefined_map": deepcopy(predefined_map)},
    )

__all__ = [
    "CoverageGridworld",
    "STANDARD_MAPS",
    "STANDARD_MAP_ORDER",
    "clone_map",
    "clone_map_suite",
]
