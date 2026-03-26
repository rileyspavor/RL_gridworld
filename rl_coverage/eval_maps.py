from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from rl_coverage.evaluate import _algorithm_registry
from rl_coverage.env_factory import make_env_builder
from rl_coverage.maps import STANDARD_MAP_ORDER
from rl_coverage.metrics import evaluate_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on each standard map individually.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing config.json and model files.")
    parser.add_argument("--model", default=None, help="Optional model path. Defaults to best_model then final_model.")
    parser.add_argument("--episodes", type=int, default=12, help="Episodes per map.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to <run-dir>/map_eval.")
    return parser.parse_args(argv)


def _load_config(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text())


def _resolve_model_path(run_dir: Path, model: str | None) -> Path:
    if model is not None:
        return Path(model)
    if (run_dir / "best_model.zip").exists():
        return run_dir / "best_model"
    return run_dir / "final_model"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "map_name",
                "episodes",
                "mean_reward",
                "mean_coverage",
                "success_rate",
                "death_rate",
                "timeout_rate",
                "mean_length",
                "min_coverage",
                "max_coverage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [row["map_name"] for row in rows]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, [row["mean_coverage"] for row in rows], color="#54A24B")
    plt.ylabel("Mean coverage")
    plt.title("Per-map mean coverage")
    plt.tight_layout()
    plt.savefig(output_dir / "per_map_coverage.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    x = range(len(rows))
    plt.bar(x, [row["success_rate"] for row in rows], width=0.4, label="success", color="#4C78A8")
    plt.bar([i + 0.4 for i in x], [row["death_rate"] for row in rows], width=0.4, label="death", color="#E45756")
    plt.xticks([i + 0.2 for i in x], labels)
    plt.ylabel("Rate")
    plt.title("Per-map success vs death")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_map_outcomes.png", dpi=180, bbox_inches="tight")
    plt.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "map_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(run_dir)
    model_path = _resolve_model_path(run_dir, args.model)
    algorithm_name = config["algorithm"]["name"]
    model_cls = _algorithm_registry()[algorithm_name]
    model = model_cls.load(model_path)

    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "model_path": str(model_path),
        "episodes_per_map": args.episodes,
        "maps": {},
    }

    for map_name in STANDARD_MAP_ORDER:
        map_config = deepcopy(config)
        map_config["environment"]["id"] = map_name
        map_config["environment"]["map_suite"] = None
        evaluation = evaluate_model(
            model,
            env_builder=make_env_builder(map_config),
            episodes=args.episodes,
            deterministic=bool(config.get("evaluation", {}).get("deterministic", True)),
        )
        summary = dict(evaluation["summary"])
        summary["map_name"] = map_name
        rows.append(summary)
        payload["maps"][map_name] = evaluation
        print(
            f"{map_name}: coverage={summary['mean_coverage']:.3f} | success={summary['success_rate']:.3f} | "
            f"death={summary['death_rate']:.3f} | reward={summary['mean_reward']:.3f}"
        )

    (output_dir / "per_map_evaluation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_csv(rows, output_dir / "per_map_summary.csv")
    _plot(rows, output_dir)
    print(f"Wrote per-map evaluation artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
