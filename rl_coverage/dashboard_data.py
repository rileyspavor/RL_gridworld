from __future__ import annotations

import csv
import hashlib
import json
import shlex
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from rl_coverage.config import load_config

RUN_MARKER_FILES = {
    "config.json",
    "summary.json",
    "final_evaluation.json",
    "evaluations.json",
    "latest_evaluation.json",
    "best_model_metrics.json",
}

JSON_ARTIFACT_NAMES = {
    "config.json": "run_config",
    "summary.json": "final_summary",
    "final_evaluation.json": "final_evaluation",
    "latest_evaluation.json": "latest_evaluation",
    "evaluations.json": "evaluation_history",
    "best_model_metrics.json": "best_checkpoint_metrics",
    "manual_evaluation.json": "manual_evaluation",
    "manual_evaluation_summary.json": "manual_summary",
    "per_map_evaluation.json": "per_map_evaluation",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class QueueJobRequest:
    config_id: str | None = None
    config_path: str | None = None
    seed: int | None = None
    output_dir: str | None = None
    notes: str = ""
    extra_args: list[str] | None = None


class ProjectDashboardService:
    def __init__(
        self,
        project_root: str | Path | None = None,
        configs_dir: str = "configs",
        runs_dir: str = "runs",
        results_dir: str = "results",
        queue_dir: str = "dashboard_queue",
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.configs_dir = self.project_root / configs_dir
        self.runs_dir = self.project_root / runs_dir
        self.results_dir = self.project_root / results_dir
        self.queue_dir = self.project_root / queue_dir

        self._catalog: dict[str, Any] | None = None
        self._config_by_id: dict[str, dict[str, Any]] = {}
        self._config_by_resolved_path: dict[str, dict[str, Any]] = {}
        self._run_dirs_by_id: dict[str, Path] = {}

    # -----------------
    # public API
    # -----------------
    def snapshot(self, refresh: bool = True) -> dict[str, Any]:
        catalog = self._refresh_catalog() if refresh or self._catalog is None else self._catalog
        assert catalog is not None
        return {
            "generated_at": utc_now_iso(),
            "project_root": str(self.project_root),
            "paths": {
                "configs": self._relative_or_none(self.configs_dir),
                "runs": self._relative_or_none(self.runs_dir),
                "results": self._relative_or_none(self.results_dir),
                "queue": self._relative_or_none(self.queue_dir),
            },
            "counts": {
                "configs": len(catalog["configs"]),
                "runs": len(catalog["runs"]),
                "result_collections": len(catalog["results"]),
                "queue_jobs": len(catalog["queue"]["jobs"]),
            },
            "leaderboard": catalog["leaderboard"],
            "configs": catalog["configs"],
            "runs": catalog["runs"],
            "results": catalog["results"],
            "queue": catalog["queue"],
        }

    def list_runs(self, refresh: bool = True) -> list[dict[str, Any]]:
        return self.snapshot(refresh=refresh)["runs"]

    def get_run(self, run_id: str, refresh: bool = True) -> dict[str, Any] | None:
        if refresh or self._catalog is None:
            self._refresh_catalog()
        run_dir = self._run_dirs_by_id.get(run_id)
        if run_dir is None:
            return None
        return self._build_run_record(run_dir, detailed=True)

    def list_configs(self, refresh: bool = True) -> list[dict[str, Any]]:
        return self.snapshot(refresh=refresh)["configs"]

    def get_config(self, config_id: str, refresh: bool = True) -> dict[str, Any] | None:
        if refresh or self._catalog is None:
            self._refresh_catalog()
        config = self._config_by_id.get(config_id)
        if config is None:
            return None
        return config

    def list_results(self, refresh: bool = True) -> list[dict[str, Any]]:
        return self.snapshot(refresh=refresh)["results"]

    def queue_state(self, refresh: bool = True) -> dict[str, Any]:
        return self.snapshot(refresh=refresh)["queue"]

    def enqueue_job(self, request: QueueJobRequest | dict[str, Any]) -> dict[str, Any]:
        if isinstance(request, dict):
            request = QueueJobRequest(**request)
        self._refresh_catalog()

        config = None
        if request.config_id:
            config = self._config_by_id.get(request.config_id)
        elif request.config_path:
            resolved = self._resolve_from_user_path(request.config_path)
            config = self._config_by_resolved_path.get(str(resolved))
            if config is None and resolved.is_file():
                config = self._build_config_record(resolved)
                self._config_by_id[config["id"]] = config
                self._config_by_resolved_path[str(resolved)] = config

        if config is None:
            raise ValueError("Queue job requires a valid config_id or config_path.")

        created_at = utc_now_iso()
        config_rel = config["relative_path"]
        config_name = config["name"]
        digest = hashlib.sha1(f"{created_at}:{config_rel}:{request.seed}:{request.output_dir}".encode()).hexdigest()[:12]
        job_id = f"job-{config_name}-{digest}"
        command = self.build_train_command(
            config_path=config_rel,
            seed=request.seed,
            output_dir=request.output_dir,
            extra_args=request.extra_args or [],
        )

        payload = {
            "id": job_id,
            "status": "pending",
            "created_at": created_at,
            "config_id": config["id"],
            "config_name": config_name,
            "config_path": config_rel,
            "seed": request.seed,
            "output_dir": request.output_dir,
            "notes": request.notes,
            "extra_args": request.extra_args or [],
            "command": command,
        }

        pending_dir = self.queue_dir / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        job_path = pending_dir / f"{job_id}.json"
        job_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return self._build_queue_job_record(job_path, status="pending")

    def build_train_command(
        self,
        config_path: str,
        seed: int | None = None,
        output_dir: str | None = None,
        extra_args: list[str] | None = None,
    ) -> dict[str, Any]:
        argv = ["python", "-m", "rl_coverage.train", "--config", config_path]
        if output_dir:
            argv.extend(["--output-dir", output_dir])
        if seed is not None:
            argv.extend(["--seed", str(seed)])
        for item in extra_args or []:
            argv.append(str(item))
        return {
            "argv": argv,
            "shell": " ".join(shlex.quote(part) for part in argv),
        }

    # -----------------
    # catalog builders
    # -----------------
    def _refresh_catalog(self) -> dict[str, Any]:
        configs = self._collect_configs()
        runs = self._collect_runs()
        results = self._collect_results()
        queue = self._collect_queue_state()
        leaderboard = self._build_leaderboard(runs)

        self._catalog = {
            "configs": configs,
            "runs": runs,
            "results": results,
            "queue": queue,
            "leaderboard": leaderboard,
        }
        return self._catalog

    def _collect_configs(self) -> list[dict[str, Any]]:
        configs: list[dict[str, Any]] = []
        self._config_by_id = {}
        self._config_by_resolved_path = {}

        if not self.configs_dir.exists():
            return configs

        for path in sorted(self.configs_dir.rglob("*.toml")):
            if path.name.startswith("."):
                continue
            record = self._build_config_record(path)
            configs.append(record)
            self._config_by_id[record["id"]] = record
            self._config_by_resolved_path[str(path.resolve())] = record

        return configs

    def _collect_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        self._run_dirs_by_id = {}

        if not self.runs_dir.exists():
            return runs

        for run_dir in self._discover_run_dirs():
            record = self._build_run_record(run_dir, detailed=False)
            runs.append(record)
            self._run_dirs_by_id[record["id"]] = run_dir

        runs.sort(
            key=lambda item: (
                item.get("metrics", {}).get("mean_coverage", 0.0),
                item.get("metrics", {}).get("success_rate", 0.0),
                item.get("created_at") or "",
            ),
            reverse=True,
        )

        linked_runs: dict[str, list[str]] = {}
        for run in runs:
            config_id = run.get("config_id")
            if config_id:
                linked_runs.setdefault(config_id, []).append(run["id"])

        for config_id, run_ids in linked_runs.items():
            config = self._config_by_id.get(config_id)
            if config is not None:
                config["linked_run_ids"] = run_ids

        return runs

    def _collect_results(self) -> list[dict[str, Any]]:
        collections: dict[str, dict[str, Any]] = {}
        if not self.results_dir.exists():
            return []

        for path in sorted(self.results_dir.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            group_rel = self._relative_or_none(path.parent)
            assert group_rel is not None
            record = collections.setdefault(
                group_rel,
                {
                    "id": self._stable_id("results", group_rel),
                    "name": path.parent.name if path.parent != self.results_dir else "results",
                    "relative_path": group_rel,
                    "file_count": 0,
                    "kinds": {},
                    "files": [],
                    "previews": [],
                },
            )
            artifact = self._artifact_record(path, root=self.project_root)
            record["files"].append(artifact)
            record["file_count"] += 1
            kinds = Counter(record["kinds"])
            kinds[artifact["kind"]] += 1
            record["kinds"] = dict(kinds)
            preview = self._preview_artifact(path)
            if preview is not None:
                record["previews"].append(preview)

        result_list = list(collections.values())
        result_list.sort(key=lambda item: item["relative_path"])
        return result_list

    def _collect_queue_state(self) -> dict[str, Any]:
        jobs: list[dict[str, Any]] = []
        statuses = ("pending", "running", "completed", "failed")
        for status in statuses:
            status_dir = self.queue_dir / status
            if not status_dir.exists():
                continue
            for job_path in sorted(status_dir.glob("*.json")):
                jobs.append(self._build_queue_job_record(job_path, status=status))
        jobs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return {
            "root": self._relative_or_none(self.queue_dir),
            "jobs": jobs,
        }

    def _build_leaderboard(self, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for run in runs:
            metrics = run.get("metrics") or {}
            if not metrics:
                continue
            items.append(
                {
                    "run_id": run["id"],
                    "name": run["name"],
                    "relative_path": run["relative_path"],
                    "group": run["group"],
                    "algorithm": run.get("algorithm"),
                    "observation": run.get("observation"),
                    "reward": run.get("reward"),
                    "environment": run.get("environment"),
                    "map_suite": run.get("map_suite"),
                    **metrics,
                }
            )
        items.sort(
            key=lambda item: (
                item.get("mean_coverage", 0.0),
                item.get("success_rate", 0.0),
                -item.get("death_rate", 0.0),
            ),
            reverse=True,
        )
        return items

    # -----------------
    # config parsing
    # -----------------
    def _build_config_record(self, path: Path) -> dict[str, Any]:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        resolved = load_config(path)
        rel_path = self._relative_or_none(path)
        assert rel_path is not None

        config_id = self._stable_id("cfg", rel_path)
        experiment = resolved.get("experiment", {})
        launch = self.build_train_command(config_path=rel_path)
        output_root = experiment.get("output_root", "runs")

        return {
            "id": config_id,
            "name": experiment.get("name") or path.stem,
            "relative_path": rel_path,
            "category": "/".join(Path(rel_path).parts[1:-1]) or "root",
            "experiment": experiment,
            "algorithm": resolved.get("algorithm", {}),
            "environment": resolved.get("environment", {}),
            "observation": resolved.get("observation", {}),
            "reward": resolved.get("reward", {}),
            "training": resolved.get("training", {}),
            "evaluation": resolved.get("evaluation", {}),
            "raw": raw,
            "resolved": resolved,
            "launch": {
                **launch,
                "suggested_output_root": output_root,
            },
            "linked_run_ids": [],
        }

    # -----------------
    # run parsing
    # -----------------
    def _discover_run_dirs(self) -> list[Path]:
        run_dirs: list[Path] = []
        seen: set[Path] = set()
        for path in sorted(self.runs_dir.rglob("*")):
            if not path.is_dir():
                continue
            marker_names = {child.name for child in path.iterdir() if child.is_file()}
            has_marker = bool(marker_names & RUN_MARKER_FILES) or any(
                name.startswith("manual_evaluation") and name.endswith(".json") for name in marker_names
            )
            if has_marker and path not in seen:
                run_dirs.append(path)
                seen.add(path)
        return run_dirs

    def _build_run_record(self, run_dir: Path, detailed: bool) -> dict[str, Any]:
        rel_path = self._relative_or_none(run_dir)
        assert rel_path is not None
        run_id = self._stable_id("run", rel_path)

        config_payload = self._read_json(run_dir / "config.json") or {}
        summary_payload = self._read_json(run_dir / "summary.json") or {}
        final_evaluation = self._read_json(run_dir / "final_evaluation.json") or {}
        latest_evaluation = self._read_json(run_dir / "latest_evaluation.json") or {}
        best_checkpoint = self._read_json(run_dir / "best_model_metrics.json") or {}
        evaluations = self._read_json(run_dir / "evaluations.json") or {}
        manual_evaluation = self._read_json(run_dir / "manual_evaluation.json") or {}
        manual_summary = self._read_json(run_dir / "manual_evaluation_summary.json") or {}

        metrics = self._resolve_run_metrics(summary_payload, final_evaluation, latest_evaluation)
        learning_curve = self._extract_learning_curve(evaluations)
        best_curve_point = self._best_curve_point(learning_curve)
        if best_curve_point is None and best_checkpoint:
            best_curve_point = {
                "timesteps": best_checkpoint.get("timesteps"),
                **(self._summary_block(best_checkpoint) or {}),
            }
        seed = self._nested_get(config_payload, ["experiment", "seed"])
        notes = self._nested_get(config_payload, ["experiment", "notes"])
        created_at = self._iso_mtime(run_dir)
        group = self._run_group(rel_path)
        config_id = self._resolve_config_id(config_payload, summary_payload)

        record: dict[str, Any] = {
            "id": run_id,
            "name": run_dir.name,
            "relative_path": rel_path,
            "group": group,
            "status": self._run_status(run_dir, summary_payload, final_evaluation),
            "created_at": created_at,
            "seed": seed,
            "notes": notes,
            "config_id": config_id,
            "config_path": self._resolved_config_relative_path(config_payload, summary_payload),
            "algorithm": summary_payload.get("algorithm") or self._nested_get(config_payload, ["algorithm", "name"]),
            "observation": summary_payload.get("observation") or self._nested_get(config_payload, ["observation", "name"]),
            "reward": summary_payload.get("reward") or self._nested_get(config_payload, ["reward", "name"]),
            "environment": summary_payload.get("environment") or self._nested_get(config_payload, ["environment", "id"]),
            "map_suite": summary_payload.get("map_suite") or self._nested_get(config_payload, ["environment", "map_suite"]),
            "metrics": metrics,
            "latest_metrics": self._summary_block(latest_evaluation),
            "best_checkpoint": best_curve_point,
            "manual_metrics": self._summary_block(manual_evaluation) or manual_summary or None,
            "artifact_counts": self._artifact_counts(run_dir),
        }

        if detailed:
            artifacts = [self._artifact_record(path, root=run_dir) for path in self._iter_artifact_files(run_dir)]
            record.update(
                {
                    "config": config_payload,
                    "summary": summary_payload,
                    "final_evaluation": final_evaluation,
                    "latest_evaluation": latest_evaluation,
                    "best_model_metrics": best_checkpoint,
                    "manual_evaluation": manual_evaluation,
                    "manual_evaluation_summary": manual_summary or self._summary_block(manual_evaluation),
                    "learning_curve": learning_curve,
                    "per_map_evaluations": self._collect_per_map_evaluations(run_dir),
                    "artifacts": artifacts,
                }
            )
        else:
            record["detail_url"] = f"/api/runs/{quote(run_id)}"

        return record

    def _resolve_run_metrics(
        self,
        summary_payload: dict[str, Any],
        final_evaluation: dict[str, Any],
        latest_evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        if summary_payload:
            summary = dict(summary_payload)
            for key in ("run_dir", "config_path", "algorithm", "observation", "reward", "environment", "map_suite"):
                summary.pop(key, None)
            return summary
        return self._summary_block(final_evaluation) or self._summary_block(latest_evaluation) or {}

    def _summary_block(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        if not payload:
            return {}
        summary = payload.get("summary")
        if isinstance(summary, dict):
            return summary
        episodes = payload.get("episodes")
        if isinstance(episodes, list):
            return self._summarize_episodes(episodes)
        return {}

    def _extract_learning_curve(self, evaluations: dict[str, Any]) -> list[dict[str, Any]]:
        items = evaluations.get("items")
        if not isinstance(items, list):
            return []
        curve: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            summary = item.get("summary")
            if not isinstance(summary, dict):
                summary = self._summarize_episodes(item.get("episodes") or [])
            point = {
                "timesteps": item.get("timesteps"),
                **summary,
            }
            curve.append(point)
        return curve

    def _best_curve_point(self, curve: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not curve:
            return None
        return max(
            curve,
            key=lambda item: (
                item.get("mean_coverage", 0.0),
                item.get("success_rate", 0.0),
                -item.get("death_rate", 0.0),
                item.get("timesteps", 0),
            ),
        )

    def _collect_per_map_evaluations(self, run_dir: Path) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for candidate in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("map_eval")):
            eval_path = candidate / "per_map_evaluation.json"
            payload = self._read_json(eval_path)
            if not payload:
                continue
            maps_payload = payload.get("maps") or {}
            maps: list[dict[str, Any]] = []
            for map_name, entry in maps_payload.items():
                if not isinstance(entry, dict):
                    continue
                summary = entry.get("summary")
                if not isinstance(summary, dict):
                    summary = self._summarize_episodes(entry.get("episodes") or [])
                maps.append(
                    {
                        "name": map_name,
                        "summary": summary,
                    }
                )
            plots = [
                self._artifact_record(path, root=self.project_root)
                for path in sorted(candidate.iterdir())
                if path.is_file() and path.suffix.lower() in {".png", ".csv", ".json"}
            ]
            outputs.append(
                {
                    "label": candidate.name,
                    "relative_path": self._relative_or_none(candidate),
                    "episodes_per_map": payload.get("episodes_per_map"),
                    "maps": maps,
                    "artifacts": plots,
                }
            )
        return outputs

    def _artifact_counts(self, run_dir: Path) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for path in self._iter_artifact_files(run_dir):
            counter[self._artifact_kind(path)] += 1
        return dict(counter)

    def _iter_artifact_files(self, root: Path) -> list[Path]:
        return [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and not path.name.startswith(".") and path.name != ".DS_Store"
        ]

    # -----------------
    # queue parsing
    # -----------------
    def _build_queue_job_record(self, job_path: Path, status: str) -> dict[str, Any]:
        payload = self._read_json(job_path) or {}
        rel_path = self._relative_or_none(job_path)
        return {
            "id": payload.get("id") or self._stable_id("job", rel_path or str(job_path)),
            "status": status,
            "relative_path": rel_path,
            "created_at": payload.get("created_at") or self._iso_mtime(job_path),
            "config_id": payload.get("config_id"),
            "config_name": payload.get("config_name"),
            "config_path": payload.get("config_path"),
            "seed": payload.get("seed"),
            "output_dir": payload.get("output_dir"),
            "notes": payload.get("notes"),
            "extra_args": payload.get("extra_args") or [],
            "command": payload.get("command"),
            "payload": payload,
        }

    # -----------------
    # helpers
    # -----------------
    def _resolve_config_id(self, config_payload: dict[str, Any], summary_payload: dict[str, Any]) -> str | None:
        config_rel = self._resolved_config_relative_path(config_payload, summary_payload)
        if config_rel is None:
            return None
        resolved = (self.project_root / config_rel).resolve()
        config = self._config_by_resolved_path.get(str(resolved))
        return config["id"] if config else None

    def _resolved_config_relative_path(self, config_payload: dict[str, Any], summary_payload: dict[str, Any]) -> str | None:
        config_path = config_payload.get("_config_path") or summary_payload.get("config_path")
        if not config_path:
            return None
        try:
            return self._relative_or_none(Path(config_path).resolve())
        except FileNotFoundError:
            return None

    def _run_group(self, rel_path: str) -> str:
        parts = Path(rel_path).parts
        if len(parts) <= 2:
            return "default"
        return parts[1]

    def _run_status(self, run_dir: Path, summary_payload: dict[str, Any], final_evaluation: dict[str, Any]) -> str:
        if summary_payload or final_evaluation:
            return "completed"
        if (run_dir / "latest_evaluation.json").exists():
            return "in_progress"
        return "partial"

    def _stable_id(self, prefix: str, value: str) -> str:
        slug = Path(value).stem.replace(" ", "-") or prefix
        digest = hashlib.sha1(value.encode()).hexdigest()[:12]
        return f"{prefix}-{slug}-{digest}"

    def _relative_or_none(self, path: Path) -> str | None:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return None

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _resolve_from_user_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def _nested_get(self, payload: dict[str, Any], keys: list[str]) -> Any:
        current: Any = payload
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _iso_mtime(self, path: Path) -> str:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )

    def _summarize_episodes(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        if not episodes:
            return {}
        coverages = [self._as_float(item.get("coverage_ratio")) for item in episodes if item.get("coverage_ratio") is not None]
        rewards = [self._as_float(item.get("total_reward")) for item in episodes if item.get("total_reward") is not None]
        lengths = [self._as_float(item.get("episode_length")) for item in episodes if item.get("episode_length") is not None]
        total = len(episodes)
        successes = sum(1 for item in episodes if bool(item.get("success")))
        timeouts = sum(1 for item in episodes if bool(item.get("timeout")))
        deaths = sum(1 for item in episodes if not bool(item.get("success")) and not bool(item.get("timeout")))
        return {
            "episodes": total,
            "mean_coverage": self._mean(coverages),
            "min_coverage": min(coverages) if coverages else None,
            "max_coverage": max(coverages) if coverages else None,
            "mean_reward": self._mean(rewards),
            "mean_length": self._mean(lengths),
            "success_rate": successes / total if total else 0.0,
            "timeout_rate": timeouts / total if total else 0.0,
            "death_rate": deaths / total if total else 0.0,
        }

    def _mean(self, values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    def _as_float(self, value: Any) -> float:
        return float(value)

    def _artifact_record(self, path: Path, root: Path) -> dict[str, Any]:
        relative_path = self._relative_or_none(path) or path.relative_to(root).as_posix()
        return {
            "name": path.name,
            "relative_path": relative_path,
            "file_url": f"/artifacts/{quote(relative_path)}",
            "kind": self._artifact_kind(path),
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "modified_at": self._iso_mtime(path),
        }

    def _artifact_kind(self, path: Path) -> str:
        if "events.out.tfevents" in path.name:
            return "tensorboard"
        if path.name in JSON_ARTIFACT_NAMES:
            return JSON_ARTIFACT_NAMES[path.name]
        if path.suffix.lower() == ".zip":
            return "model"
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return "plot"
        if path.suffix.lower() == ".csv":
            return "table"
        if path.suffix.lower() == ".md":
            return "notes"
        if path.suffix.lower() == ".json":
            return "json"
        return "other"

    def _preview_artifact(self, path: Path) -> dict[str, Any] | None:
        suffix = path.suffix.lower()
        rel_path = self._relative_or_none(path)
        if suffix == ".json" and path.stat().st_size <= 250_000:
            payload = self._read_json(path)
            if payload is None:
                return None
            preview: dict[str, Any] = {
                "relative_path": rel_path,
                "type": "json",
            }
            if isinstance(payload, dict):
                preview["keys"] = sorted(payload.keys())
                if isinstance(payload.get("summary"), dict):
                    preview["summary"] = payload["summary"]
                if isinstance(payload.get("conclusions"), list):
                    preview["conclusions"] = payload["conclusions"][:3]
            return preview
        if suffix == ".csv":
            try:
                with path.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    rows = []
                    for index, row in enumerate(reader):
                        rows.append(row)
                        if index >= 4:
                            break
                return {
                    "relative_path": rel_path,
                    "type": "csv",
                    "rows": rows,
                }
            except OSError:
                return None
        if suffix == ".md" and path.stat().st_size <= 40_000:
            try:
                text = path.read_text()
            except OSError:
                return None
            return {
                "relative_path": rel_path,
                "type": "markdown",
                "excerpt": text[:600],
            }
        return None
