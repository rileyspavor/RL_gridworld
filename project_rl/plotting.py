"""Reporting utilities for experiment-run comparison artifacts.

This module reads per-run summary/evaluation JSON files and produces a leaderboard
CSV plus visual comparisons used in the assignment report.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from project_rl.config import load_json


def _require_matplotlib():
    """Raise a clear error when plotting dependencies are unavailable."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to generate plots.") from exc


def load_run_records(run_dirs: list[str | Path]) -> list[dict[str, Any]]:
    """Load summary and evaluation records from run directories.

    Args:
        run_dirs: Candidate run directories containing ``summary.json`` files.

    Returns:
        List of summary dictionaries enriched with run metadata.
    """
    records: list[dict[str, Any]] = []

    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue

        summary = load_json(summary_path)
        evaluations_path = run_dir / "evaluations.json"
        evaluations = []
        if evaluations_path.exists():
            evaluations = load_json(evaluations_path).get("items", [])

        summary["_run_name"] = run_dir.name
        summary["_run_dir"] = str(run_dir)
        summary["_evaluations"] = evaluations
        records.append(summary)

    return records


def write_leaderboard_csv(records: list[dict[str, Any]], output_dir: str | Path) -> Path:
    """Write a coverage-sorted leaderboard CSV for experiment runs.

    Args:
        records: Loaded run records from :func:`load_run_records`.
        output_dir: Directory for output artifacts.

    Returns:
        Path to the written ``leaderboard.csv`` file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "leaderboard.csv"

    ordered = sorted(records, key=lambda item: item.get("mean_coverage", 0.0), reverse=True)
    fieldnames = [
        "run_name",
        "algorithm",
        "observation",
        "reward",
        "mean_coverage",
        "success_rate",
        "death_rate",
        "mean_reward",
        "mean_length",
        "run_dir",
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in ordered:
            writer.writerow(
                {
                    "run_name": record.get("_run_name"),
                    "algorithm": record.get("algorithm"),
                    "observation": record.get("observation"),
                    "reward": record.get("reward"),
                    "mean_coverage": record.get("mean_coverage"),
                    "success_rate": record.get("success_rate"),
                    "death_rate": record.get("death_rate"),
                    "mean_reward": record.get("mean_reward"),
                    "mean_length": record.get("mean_length"),
                    "run_dir": record.get("run_dir"),
                }
            )

    return csv_path


def plot_metric_bars(
    records: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    output_path: str | Path,
    dpi: int = 180,
) -> None:
    """Create a bar chart comparing runs for one scalar metric.

    Args:
        records: Run records with metric values.
        metric: Dictionary key to plot.
        ylabel: Human-readable axis label.
        output_path: Destination image path.
        dpi: Image resolution.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    ordered = sorted(records, key=lambda item: item.get(metric, 0.0), reverse=True)
    labels = [f"{row['observation']}\n{row['reward']}" for row in ordered]
    values = [float(row.get(metric, 0.0)) for row in ordered]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(ordered)), values, color="#4C78A8")
    plt.xticks(range(len(ordered)), labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by observation/reward combination")

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_learning_curves(records: list[dict[str, Any]], output_path: str | Path, dpi: int = 180) -> None:
    """Plot mean-coverage learning curves from periodic evaluations.

    Args:
        records: Run records that may include ``_evaluations`` history.
        output_path: Destination image path.
        dpi: Image resolution.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 7))
    plotted_any = False
    for record in records:
        evaluations = record.get("_evaluations", [])
        if not evaluations:
            continue

        x_values = [item["timesteps"] for item in evaluations]
        y_values = [item["summary"]["mean_coverage"] for item in evaluations]
        label = f"{record['observation']} + {record['reward']}"
        plt.plot(x_values, y_values, linewidth=1.4, label=label)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    plt.xscale("log")
    plt.xlabel("Training timesteps (log scale)")
    plt.ylabel("Evaluation mean coverage")
    plt.title("Learning curves")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_coverage_heatmap(records: list[dict[str, Any]], output_path: str | Path, dpi: int = 180) -> None:
    """Render a heatmap of mean coverage by observation/reward pairing.

    Args:
        records: Run records containing observation, reward, and coverage fields.
        output_path: Destination image path.
        dpi: Image resolution.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    observations = sorted({record["observation"] for record in records})
    rewards = sorted({record["reward"] for record in records})
    if not observations or not rewards:
        return

    matrix = np.full((len(observations), len(rewards)), np.nan, dtype=np.float32)
    for record in records:
        obs_index = observations.index(record["observation"])
        reward_index = rewards.index(record["reward"])
        matrix[obs_index, reward_index] = float(record.get("mean_coverage", 0.0))

    fig, axis = plt.subplots(figsize=(7, 4.5))
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(rewards)))
    axis.set_xticklabels(rewards, rotation=20, ha="right")
    axis.set_yticks(range(len(observations)))
    axis.set_yticklabels(observations)
    axis.set_title("Mean coverage heatmap")
    axis.set_xlabel("Reward variant")
    axis.set_ylabel("Observation variant")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "--" if np.isnan(value) else f"{value:.3f}"
            axis.text(col, row, text, ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(image, ax=axis, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_experiment_report(run_dirs: list[str | Path], output_dir: str | Path) -> list[Path]:
    """Generate all report artifacts for a set of run directories.

    Args:
        run_dirs: Run directories to include in the comparison report.
        output_dir: Directory where report files are written.

    Returns:
        List of generated file paths.

    Raises:
        RuntimeError: If no valid run summaries are found.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = load_run_records(run_dirs)
    if not records:
        raise RuntimeError("No run summaries found for report generation.")

    written_files = [write_leaderboard_csv(records, output_path)]

    coverage_path = output_path / "mean_coverage_by_combo.png"
    success_path = output_path / "success_rate_by_combo.png"
    reward_path = output_path / "mean_reward_by_combo.png"
    curves_path = output_path / "learning_curves.png"
    heatmap_path = output_path / "coverage_heatmap.png"

    plot_metric_bars(records, metric="mean_coverage", ylabel="Mean coverage", output_path=coverage_path)
    plot_metric_bars(records, metric="success_rate", ylabel="Success rate", output_path=success_path)
    plot_metric_bars(records, metric="mean_reward", ylabel="Mean reward", output_path=reward_path)
    plot_learning_curves(records, output_path=curves_path)
    plot_coverage_heatmap(records, output_path=heatmap_path)

    written_files.extend([coverage_path, success_path, reward_path, curves_path, heatmap_path])
    return written_files
