from __future__ import annotations

from statistics import mean
from typing import Any

import numpy as np


def summarize_episode(final_info: dict[str, Any], total_reward: float, length: int) -> dict[str, Any]:
    coverable = max(1, int(final_info["coverable_cells"]))
    covered = int(final_info["total_covered_cells"])
    success = bool(final_info.get("success", final_info["cells_remaining"] == 0))
    timeout = bool(final_info.get("timeout", False))
    game_over = bool(final_info["game_over"])
    return {
        "total_reward": float(total_reward),
        "episode_length": int(length),
        "covered_cells": covered,
        "coverable_cells": coverable,
        "coverage_ratio": covered / coverable,
        "cells_remaining": int(final_info["cells_remaining"]),
        "steps_remaining": int(final_info["steps_remaining"]),
        "success": success,
        "timeout": timeout,
        "game_over": game_over,
    }


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {
            "episodes": 0,
            "mean_reward": 0.0,
            "mean_coverage": 0.0,
            "success_rate": 0.0,
            "timeout_rate": 0.0,
            "death_rate": 0.0,
            "mean_length": 0.0,
        }

    return {
        "episodes": len(episodes),
        "mean_reward": mean(ep["total_reward"] for ep in episodes),
        "mean_coverage": mean(ep["coverage_ratio"] for ep in episodes),
        "success_rate": mean(float(ep["success"]) for ep in episodes),
        "timeout_rate": mean(float(ep["timeout"]) for ep in episodes),
        "death_rate": mean(float(ep["game_over"]) for ep in episodes),
        "mean_length": mean(ep["episode_length"] for ep in episodes),
        "min_coverage": min(ep["coverage_ratio"] for ep in episodes),
        "max_coverage": max(ep["coverage_ratio"] for ep in episodes),
    }


def evaluate_model(model, env_builder, episodes: int, deterministic: bool = True, seed: int | None = None):
    env = env_builder()
    summaries: list[dict[str, Any]] = []

    try:
        for episode_index in range(episodes):
            reset_kwargs = {}
            if seed is not None:
                reset_kwargs["seed"] = seed + episode_index
            observation, _ = env.reset(**reset_kwargs)
            done = False
            total_reward = 0.0
            length = 0
            final_info = None

            while not done:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                length += 1
                done = bool(terminated or truncated)
                if done:
                    final_info = info

            assert final_info is not None
            summaries.append(summarize_episode(final_info, total_reward=total_reward, length=length))
    finally:
        env.close()

    return {
        "episodes": summaries,
        "summary": aggregate_episodes(summaries),
    }


def evaluation_text(summary: dict[str, Any]) -> str:
    return (
        f"episodes={summary['episodes']} | mean_reward={summary['mean_reward']:.3f} | "
        f"mean_coverage={summary['mean_coverage']:.3f} | success_rate={summary['success_rate']:.3f} | "
        f"death_rate={summary['death_rate']:.3f} | timeout_rate={summary['timeout_rate']:.3f} | "
        f"mean_length={summary['mean_length']:.1f}"
    )
