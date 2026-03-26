from __future__ import annotations

import argparse
import csv
import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from PIL import Image
import pygame

from rl_coverage.evaluate import _algorithm_registry
from rl_coverage.env_factory import make_env
from rl_coverage.maps import STANDARD_MAP_ORDER
from rl_coverage.metrics import summarize_episode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic replay GIFs for a trained Coverage Gridworld model.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing config.json and model files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write replay artifacts into.")
    parser.add_argument(
        "--maps",
        nargs="+",
        default=list(STANDARD_MAP_ORDER),
        help="Maps to search/render. Defaults to all standard maps.",
    )
    parser.add_argument(
        "--focus-maps",
        nargs="*",
        default=["chokepoint", "sneaky_enemies"],
        help="Maps that also get a best-of-search replay in addition to the representative replay.",
    )
    parser.add_argument("--search-seeds", type=int, default=16, help="Number of seeds to scan per map before selecting replays.")
    parser.add_argument("--frame-stride", type=int, default=4, help="Keep every Nth environment step in the exported GIF.")
    parser.add_argument("--gif-ms", type=int, default=120, help="Milliseconds per GIF frame.")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic policy actions.")
    return parser.parse_args(argv)


def _load_config(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text())


def _resolve_model_path(run_dir: Path) -> Path:
    if (run_dir / "best_model.zip").exists():
        return run_dir / "best_model"
    return run_dir / "final_model"


def _load_model(config: dict[str, Any], model_path: Path):
    model_cls = _algorithm_registry()[config["algorithm"]["name"]]
    return model_cls.load(model_path)


def _single_map_config(base_config: dict[str, Any], map_name: str, render: bool) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["environment"]["map_suite"] = None
    config["environment"]["id"] = map_name
    config["environment"]["render_mode"] = "human" if render else None
    return config


def _surface_to_image(surface: pygame.Surface) -> Image.Image:
    array = pygame.surfarray.array3d(surface)
    return Image.fromarray(array.swapaxes(0, 1))


def rollout(
    model,
    base_config: dict[str, Any],
    map_name: str,
    seed: int,
    deterministic: bool,
    render: bool,
    frame_stride: int,
) -> tuple[dict[str, Any], list[Image.Image]]:
    random.seed(seed)
    np.random.seed(seed)
    config = _single_map_config(base_config, map_name=map_name, render=render)
    env = make_env(config, render_mode="human" if render else None)
    total_reward = 0.0
    length = 0
    frames: list[Image.Image] = []
    final_info: dict[str, Any] | None = None

    try:
        observation, _ = env.reset(seed=seed)
        if render:
            env.unwrapped.render()
            frames.append(_surface_to_image(env.unwrapped.window_surface))

        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            length += 1
            done = bool(terminated or truncated)
            if render and (length % frame_stride == 0 or done):
                frames.append(_surface_to_image(env.unwrapped.window_surface))
            if done:
                final_info = info
    finally:
        env.close()

    assert final_info is not None
    summary = summarize_episode(final_info, total_reward=total_reward, length=length)
    payload = {
        "map_name": map_name,
        "seed": seed,
        "summary": summary,
        "frame_count": len(frames),
        "frame_stride": frame_stride,
    }
    return payload, frames


def _best_key(item: dict[str, Any]) -> tuple[float, float, float, float, int]:
    summary = item["summary"]
    return (
        float(summary["coverage_ratio"]),
        float(summary["success"]),
        float(not summary["game_over"]),
        -float(summary["cells_remaining"]),
        -int(item["seed"]),
    )


def choose_representative(results: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        results,
        key=lambda item: (
            float(item["summary"]["coverage_ratio"]),
            float(item["summary"]["success"]),
            float(not item["summary"]["game_over"]),
            -int(item["seed"]),
        ),
    )
    return ordered[len(ordered) // 2]


def choose_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(results, key=_best_key)


def _write_search_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "map_name",
                "seed",
                "coverage_ratio",
                "covered_cells",
                "coverable_cells",
                "success",
                "game_over",
                "timeout",
                "episode_length",
                "cells_remaining",
                "steps_remaining",
                "total_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_gif(frames: list[Image.Image], output_path: Path, duration_ms: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    first, *rest = frames
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def export_selected_rollout(
    model,
    base_config: dict[str, Any],
    selection: dict[str, Any],
    output_dir: Path,
    label: str,
    deterministic: bool,
    frame_stride: int,
    gif_ms: int,
) -> dict[str, Any]:
    payload, frames = rollout(
        model,
        base_config=base_config,
        map_name=selection["map_name"],
        seed=int(selection["seed"]),
        deterministic=deterministic,
        render=True,
        frame_stride=frame_stride,
    )
    map_dir = output_dir / payload["map_name"]
    stem = f"{label}_seed{payload['seed']:02d}_cov{payload['summary']['coverage_ratio']:.3f}".replace(".", "p")
    gif_path = map_dir / f"{stem}.gif"
    summary_path = map_dir / f"{stem}.json"
    _save_gif(frames, gif_path, duration_ms=gif_ms)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload["gif_path"] = str(gif_path)
    payload["summary_path"] = str(summary_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = _load_config(run_dir)
    model_path = _resolve_model_path(run_dir)
    model = _load_model(base_config, model_path)

    manifest: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "model_path": str(model_path),
        "deterministic": bool(args.deterministic or base_config.get("evaluation", {}).get("deterministic", True)),
        "search_seeds": int(args.search_seeds),
        "frame_stride": int(args.frame_stride),
        "gif_ms": int(args.gif_ms),
        "maps": {},
        "exports": [],
    }

    search_rows: list[dict[str, Any]] = []
    deterministic = manifest["deterministic"]

    for map_name in args.maps:
        search_results: list[dict[str, Any]] = []
        for seed in range(args.search_seeds):
            payload, _ = rollout(
                model,
                base_config=base_config,
                map_name=map_name,
                seed=seed,
                deterministic=deterministic,
                render=False,
                frame_stride=max(1, args.frame_stride),
            )
            row = {
                "map_name": map_name,
                "seed": seed,
                **payload["summary"],
            }
            search_rows.append(row)
            search_results.append(payload)

        representative = choose_representative(search_results)
        best = choose_best(search_results)
        map_payload = {
            "search": search_results,
            "representative": representative,
            "best": best,
        }
        manifest["maps"][map_name] = map_payload

        exported = export_selected_rollout(
            model,
            base_config=base_config,
            selection=representative,
            output_dir=output_dir,
            label="representative",
            deterministic=deterministic,
            frame_stride=max(1, args.frame_stride),
            gif_ms=max(20, args.gif_ms),
        )
        manifest["exports"].append(exported)

        if map_name in set(args.focus_maps):
            if int(best["seed"]) != int(representative["seed"]):
                exported = export_selected_rollout(
                    model,
                    base_config=base_config,
                    selection=best,
                    output_dir=output_dir,
                    label="best",
                    deterministic=deterministic,
                    frame_stride=max(1, args.frame_stride),
                    gif_ms=max(20, args.gif_ms),
                )
                manifest["exports"].append(exported)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_search_csv(search_rows, output_dir / "seed_search.csv")
    print(f"Wrote replay artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
