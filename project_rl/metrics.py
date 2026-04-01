"""Evaluation helpers for per-episode and aggregate experiment metrics.

These utilities convert raw environment ``info`` payloads into stable summary
statistics used by training callbacks, CLI evaluation, and leaderboard reports.
"""

from __future__ import annotations

from statistics import mean
from typing import Any


def summarize_episode(final_info: dict[str, Any], total_reward: float, episode_length: int) -> dict[str, Any]:
    """Build a standardized metrics record for one finished episode.

    Args:
        final_info: ``info`` dictionary from the terminal environment step.
        total_reward: Sum of rewards collected during the episode.
        episode_length: Number of steps taken in the episode.

    Returns:
        Episode-level metrics including coverage, success, timeout, and death.
    """
    coverable = max(1, int(final_info["coverable_cells"]))
    covered = int(final_info["total_covered_cells"])
    success = int(final_info["cells_remaining"]) == 0
    timeout = (int(final_info["steps_remaining"]) <= 0) and (not success) and (not bool(final_info["game_over"]))
    return {
        "total_reward": float(total_reward),
        "episode_length": int(episode_length),
        "covered_cells": covered,
        "coverable_cells": coverable,
        "coverage_ratio": covered / coverable,
        "cells_remaining": int(final_info["cells_remaining"]),
        "steps_remaining": int(final_info["steps_remaining"]),
        "success": bool(success),
        "timeout": bool(timeout),
        "game_over": bool(final_info["game_over"]),
    }


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode metrics into a single summary dictionary.

    Args:
        episodes: List of per-episode records from :func:`summarize_episode`.

    Returns:
        Mean/rate metrics used throughout training and reporting.
    """
    if not episodes:
        return {
            "episodes": 0,
            "mean_reward": 0.0,
            "mean_coverage": 0.0,
            "success_rate": 0.0,
            "timeout_rate": 0.0,
            "death_rate": 0.0,
            "mean_length": 0.0,
            "min_coverage": 0.0,
            "max_coverage": 0.0,
        }

    return {
        "episodes": len(episodes),
        "mean_reward": mean(episode["total_reward"] for episode in episodes),
        "mean_coverage": mean(episode["coverage_ratio"] for episode in episodes),
        "success_rate": mean(float(episode["success"]) for episode in episodes),
        "timeout_rate": mean(float(episode["timeout"]) for episode in episodes),
        "death_rate": mean(float(episode["game_over"]) for episode in episodes),
        "mean_length": mean(episode["episode_length"] for episode in episodes),
        "min_coverage": min(episode["coverage_ratio"] for episode in episodes),
        "max_coverage": max(episode["coverage_ratio"] for episode in episodes),
    }


def evaluate_model(
    model,
    env_builder,
    episodes: int,
    deterministic: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a trained policy for multiple episodes and summarize outcomes.

    Args:
        model: Stable-Baselines model exposing ``predict``.
        env_builder: Zero-argument callable returning a fresh environment.
        episodes: Number of evaluation episodes to run.
        deterministic: Whether to sample deterministic policy actions.
        seed: Optional base seed, offset by episode index for reproducibility.

    Returns:
        Dictionary with full per-episode records and aggregate summary.
    """
    env = env_builder()
    episode_summaries: list[dict[str, Any]] = []

    try:
        for episode_index in range(episodes):
            reset_kwargs = {}
            if seed is not None:
                reset_kwargs["seed"] = seed + episode_index
            observation, _ = env.reset(**reset_kwargs)
            done = False
            total_reward = 0.0
            episode_length = 0
            final_info = None

            while not done:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                episode_length += 1
                done = bool(terminated or truncated)
                if done:
                    final_info = info

            if final_info is None:
                raise RuntimeError("Evaluation episode ended without final info payload.")

            episode_summaries.append(
                summarize_episode(
                    final_info,
                    total_reward=total_reward,
                    episode_length=episode_length,
                )
            )
    finally:
        env.close()

    return {
        "episodes": episode_summaries,
        "summary": aggregate_episodes(episode_summaries),
    }


def evaluation_text(summary: dict[str, Any]) -> str:
    """Format an aggregate evaluation summary for console logs.

    Args:
        summary: Output of :func:`aggregate_episodes`.

    Returns:
        One-line human-readable metrics string.
    """
    return (
        f"episodes={summary['episodes']} | mean_reward={summary['mean_reward']:.3f} | "
        f"mean_coverage={summary['mean_coverage']:.3f} | success_rate={summary['success_rate']:.3f} | "
        f"death_rate={summary['death_rate']:.3f} | timeout_rate={summary['timeout_rate']:.3f} | "
        f"mean_length={summary['mean_length']:.1f}"
    )
