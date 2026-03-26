from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from rl_coverage.bootstrap import *  # noqa: F401,F403
import coverage_gridworld

STANDARD_MAP_ORDER = tuple(coverage_gridworld.STANDARD_MAP_ORDER)
STANDARD_MAPS = coverage_gridworld.STANDARD_MAPS

MAP_SUITES = {
    "all_standard": list(STANDARD_MAP_ORDER),
    "easy_to_hard": ["just_go", "safe", "maze", "chokepoint", "sneaky_enemies"],
    "hard_only": ["chokepoint", "sneaky_enemies"],
    "core_generalization": ["safe", "maze", "chokepoint", "sneaky_enemies"],
    "tournament_hard_emphasis": [
        "just_go",
        "safe",
        "maze",
        "chokepoint",
        "sneaky_enemies",
        "chokepoint",
        "sneaky_enemies",
        "safe",
        "just_go",
        "sneaky_enemies",
    ],
    "tournament_hard_focus": [
        "just_go",
        "safe",
        "maze",
        "chokepoint",
        "sneaky_enemies",
        "chokepoint",
        "sneaky_enemies",
        "sneaky_enemies",
        "chokepoint",
    ],
    "tournament_safe_polish": [
        "just_go",
        "safe",
        "maze",
        "safe",
        "maze",
        "chokepoint",
        "sneaky_enemies",
        "safe",
        "chokepoint",
        "sneaky_enemies",
    ],
    "tournament_hard_anchor": [
        "just_go",
        "safe",
        "maze",
        "safe",
        "maze",
        "chokepoint",
        "sneaky_enemies",
        "chokepoint",
        "sneaky_enemies",
        "chokepoint",
        "sneaky_enemies",
        "safe",
    ],
    "tournament_sneaky_anchor": [
        "just_go",
        "safe",
        "maze",
        "safe",
        "chokepoint",
        "sneaky_enemies",
        "sneaky_enemies",
        "chokepoint",
        "sneaky_enemies",
        "safe",
        "maze",
        "sneaky_enemies",
    ],
}


def clone_map(map_name: str) -> list[list[int]]:
    return deepcopy(STANDARD_MAPS[map_name])


def clone_map_suite(map_names: Iterable[str]) -> list[list[list[int]]]:
    return [clone_map(name) for name in map_names]


def resolve_map_suite(name: str | None) -> list[list[list[int]]] | None:
    if not name:
        return None
    if name not in MAP_SUITES:
        valid = ", ".join(sorted(MAP_SUITES))
        raise KeyError(f"Unknown map suite '{name}'. Valid suites: {valid}")
    return clone_map_suite(MAP_SUITES[name])
