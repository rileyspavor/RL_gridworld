from __future__ import annotations

import csv
import json
import re
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from rl_dashboard.models import DashboardPaths, METRIC_FIELDS


PRIMARY_MAP_EVAL_PREFERENCE = (
    "map_eval_best64",
    "map_eval_best_model_32eps",
    "map_eval_best_model",
    "map_eval_final64",
    "map_eval_final_model",
    "map_eval",
    "map_eval_core_candidate",
)

ACTION_ASSET_PATH_HINTS = (
    "sanity",
    "replay",
    "playback",
    "rollout",
    "trajectory",
    "visual_summary",
    "contact_sheet",
    "contact_sheets",
    "manifest",
    "seed_search",
)

ACTION_MAP_HINTS = ("sneaky_enemies", "chokepoint", "maze", "safe", "just_go")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _read_text_preview(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines]) + "\n..."


def _dig(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _safe_number(value: str) -> Any:
    text = value.strip()
    if text == "":
        return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coverage_hint_from_path(path: str) -> float | None:
    match = re.search(r"cov(\d+(?:p\d+)?|\d+\.\d+)", path)
    if not match:
        return None
    raw = match.group(1).replace("p", ".")
    return _safe_float(raw)


def _mean_from_rows(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _family_from_run_id(run_id: str) -> str:
    parts = [part for part in run_id.split("/") if part]
    if len(parts) <= 1:
        return "baseline"
    return parts[0]


def _map_eval_rank(name: str) -> int:
    try:
        return PRIMARY_MAP_EVAL_PREFERENCE.index(name)
    except ValueError:
        return len(PRIMARY_MAP_EVAL_PREFERENCE) + 1


class DashboardRepository:
    def __init__(self, paths: DashboardPaths) -> None:
        self.paths = paths

    def scan_runs(self) -> list[dict[str, Any]]:
        runs_root = self.paths.runs_dir
        if not runs_root.exists():
            return []

        run_dirs: set[Path] = set()
        for marker in ("config.json", "summary.json", "final_evaluation.json"):
            run_dirs.update(path.parent for path in runs_root.rglob(marker))

        runs = [self._build_run_record(run_dir, include_details=False) for run_dir in sorted(run_dirs)]
        runs.sort(key=self._run_sort_key, reverse=True)
        return runs

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if not run_id:
            return None

        run_path = (self.paths.runs_dir / Path(run_id)).resolve()
        runs_root = self.paths.runs_dir.resolve()
        try:
            run_path.relative_to(runs_root)
        except ValueError:
            return None

        if not run_path.exists() or not run_path.is_dir():
            return None

        return self._build_run_record(run_path, include_details=True)

    def list_config_templates(self) -> list[dict[str, Any]]:
        configs_root = self.paths.configs_dir
        if not configs_root.exists():
            return []

        templates: list[dict[str, Any]] = []
        for config_path in sorted(configs_root.rglob("*.toml")):
            try:
                with config_path.open("rb") as handle:
                    payload = tomllib.load(handle)
            except (OSError, tomllib.TOMLDecodeError):
                continue

            rel_path = config_path.relative_to(self.paths.root).as_posix()
            templates.append(
                {
                    "path": rel_path,
                    "name": _coalesce(_dig(payload, ("experiment", "name")), config_path.stem),
                    "notes": _coalesce(_dig(payload, ("experiment", "notes")), ""),
                    "algorithm": _dig(payload, ("algorithm", "name")),
                    "observation": _dig(payload, ("observation", "name")),
                    "reward": _dig(payload, ("reward", "name")),
                    "environment": _dig(payload, ("environment", "id")),
                    "map_suite": _dig(payload, ("environment", "map_suite")),
                    "seed": _dig(payload, ("experiment", "seed")),
                    "total_timesteps": _dig(payload, ("training", "total_timesteps")),
                    "output_root": _dig(payload, ("experiment", "output_root")),
                    "policy": _dig(payload, ("algorithm", "policy")),
                }
            )
        return templates

    def load_results_bundle(self) -> dict[str, Any]:
        results_dir = self.paths.results_dir
        sweep_dir = results_dir / "observation_reward_sweep"
        leaderboard_csv = sweep_dir / "leaderboard.csv"

        sweep_leaderboard = self._read_csv_rows(leaderboard_csv)
        sweep_plots = [
            path.relative_to(self.paths.root).as_posix()
            for path in sorted(sweep_dir.glob("*.png"))
            if path.is_file()
        ]
        sweep_docs = [
            path.relative_to(self.paths.root).as_posix()
            for path in sorted(sweep_dir.glob("*"))
            if path.is_file() and path.suffix.lower() in {".csv", ".md", ".txt", ".json"}
        ]

        results_docs = [
            path.relative_to(self.paths.root).as_posix()
            for path in sorted(results_dir.glob("*"))
            if path.is_file() and path.suffix.lower() in {".csv", ".md", ".txt", ".json"}
        ]

        inbox_path = self.paths.report_dir / "EXPERIMENT_INBOX.md"
        inbox_rel = inbox_path.relative_to(self.paths.root).as_posix() if inbox_path.exists() else None

        return {
            "sweep": {
                "directory": sweep_dir.relative_to(self.paths.root).as_posix() if sweep_dir.exists() else None,
                "leaderboard": sweep_leaderboard,
                "plots": sweep_plots,
                "documents": sweep_docs,
            },
            "results_documents": results_docs,
            "experiment_inbox": {
                "path": inbox_rel,
                "preview": _read_text_preview(inbox_path, max_lines=32),
            },
        }

    def _read_csv_rows(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows.append({key: _safe_number(value or "") for key, value in row.items()})
        except OSError:
            return []
        return rows

    def _build_run_record(self, run_dir: Path, include_details: bool) -> dict[str, Any]:
        config_payload = _load_json(run_dir / "config.json")
        summary_payload = _load_json(run_dir / "summary.json")
        final_evaluation = _load_json(run_dir / "final_evaluation.json")
        latest_evaluation = _load_json(run_dir / "latest_evaluation.json")
        manual_summary = _load_json(run_dir / "manual_evaluation_summary.json")
        map_eval_summaries = self._load_map_eval_summaries(run_dir)
        primary_map_eval = self._select_primary_map_eval(map_eval_summaries)

        metrics_source = summary_payload or final_evaluation.get("summary", {}) or manual_summary

        algorithm = _coalesce(summary_payload.get("algorithm"), _dig(config_payload, ("algorithm", "name")))
        observation = _coalesce(summary_payload.get("observation"), _dig(config_payload, ("observation", "name")))
        reward = _coalesce(summary_payload.get("reward"), _dig(config_payload, ("reward", "name")))
        environment = _coalesce(summary_payload.get("environment"), _dig(config_payload, ("environment", "id")))
        map_suite = _coalesce(summary_payload.get("map_suite"), _dig(config_payload, ("environment", "map_suite")))

        status = "completed"
        if not summary_payload and not final_evaluation.get("summary"):
            status = "configured" if config_payload else "partial"

        relative_dir = run_dir.relative_to(self.paths.root).as_posix()
        run_id = run_dir.relative_to(self.paths.runs_dir).as_posix()
        run_name = run_dir.name
        family = _family_from_run_id(run_id)

        timestamp = self._parse_run_timestamp(run_name)
        config_path = self._normalize_repo_path(
            _coalesce(summary_payload.get("config_path"), config_payload.get("_config_path"))
        )
        run_dir_path = self._normalize_repo_path(summary_payload.get("run_dir")) or relative_dir

        artifact_list = self._list_run_artifacts(run_dir)
        action_assets = self._select_action_assets(artifact_list)

        record: dict[str, Any] = {
            "run_id": run_id,
            "run_name": run_name,
            "relative_dir": relative_dir,
            "run_dir": run_dir_path,
            "family": family,
            "status": status,
            "algorithm": algorithm,
            "observation": observation,
            "reward": reward,
            "environment": environment,
            "map_suite": map_suite,
            "config_path": config_path,
            "total_timesteps": _coalesce(summary_payload.get("total_timesteps"), _dig(config_payload, ("training", "total_timesteps"))),
            "timestamp": timestamp,
            "updated_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds"),
            "metrics": {metric: metrics_source.get(metric) for metric in METRIC_FIELDS},
            "artifact_count": len(artifact_list),
            "has_action_assets": bool(action_assets),
            "top_action_asset": action_assets[0] if action_assets else None,
            "has_map_eval": bool(primary_map_eval),
            "primary_map_eval": primary_map_eval,
        }

        if include_details:
            record["artifacts"] = artifact_list
            record["action_assets"] = action_assets
            record["evaluation_points"] = self._load_evaluation_points(run_dir)
            record["map_eval_summaries"] = map_eval_summaries
            record["map_eval_images"] = [item["path"] for item in artifact_list if "/map_eval" in item["path"] and item["path"].endswith(".png")]
            record["summary_payload"] = summary_payload
            record["config_payload"] = config_payload
            record["latest_evaluation_summary"] = latest_evaluation.get("summary", {})
            record["final_evaluation_summary"] = final_evaluation.get("summary", {})
            record["manual_evaluation_summary"] = manual_summary
        return record

    def _list_run_artifacts(self, run_dir: Path) -> list[dict[str, Any]]:
        allowed_suffixes = {".json", ".csv", ".png", ".gif", ".mp4", ".webm", ".md", ".txt", ".zip"}
        artifacts: list[dict[str, Any]] = []

        for file_path in sorted(run_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if "tensorboard" in file_path.parts:
                continue
            if file_path.suffix.lower() not in allowed_suffixes:
                continue
            rel_path = file_path.relative_to(self.paths.root).as_posix()
            rel_within_run = file_path.relative_to(run_dir).as_posix()
            parts = rel_within_run.split("/")
            category = parts[0] if len(parts) > 1 else "root"
            artifacts.append(
                {
                    "path": rel_path,
                    "name": file_path.name,
                    "category": category,
                    "size_bytes": file_path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
        return artifacts

    def _select_action_assets(self, artifacts: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for artifact in artifacts:
            path = str(artifact.get("path") or "")
            if not path:
                continue
            lowered = path.lower()
            suffix = Path(path).suffix.lower()
            if suffix not in {".gif", ".mp4", ".webm", ".png", ".json", ".md", ".csv"}:
                continue
            if not any(hint in lowered for hint in ACTION_ASSET_PATH_HINTS):
                continue

            map_name = self._asset_map_name(lowered)
            kind = self._action_asset_kind(lowered, suffix)
            coverage_hint = _coverage_hint_from_path(lowered)
            candidates.append(
                {
                    "path": path,
                    "name": artifact.get("name"),
                    "kind": kind,
                    "map_name": map_name,
                    "coverage_hint": coverage_hint,
                    "label": self._action_asset_label(lowered, kind, map_name, coverage_hint),
                    "score": self._action_asset_score(lowered, suffix, map_name, coverage_hint),
                }
            )

        candidates.sort(key=lambda item: (-(item.get("score") or 0), str(item.get("path") or "")))
        top_assets: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        seen_labels: set[str] = set()
        for item in candidates:
            path = str(item.get("path") or "")
            label = str(item.get("label") or "")
            if not path or path in seen_paths:
                continue
            if label and label in seen_labels:
                continue
            seen_paths.add(path)
            if label:
                seen_labels.add(label)
            top_assets.append(
                {
                    "path": path,
                    "name": item.get("name"),
                    "kind": item.get("kind"),
                    "map_name": item.get("map_name"),
                    "label": item.get("label"),
                }
            )
            if len(top_assets) >= limit:
                break
        return top_assets

    @staticmethod
    def _asset_map_name(lowered_path: str) -> str | None:
        path_obj = Path(lowered_path)
        parts = {part.lower() for part in path_obj.parts}
        filename = path_obj.name.lower()
        for map_name in ACTION_MAP_HINTS:
            if map_name in parts:
                return map_name
            if filename.startswith(f"{map_name}_") or filename.endswith(f"_{map_name}.png"):
                return map_name
        return None

    @staticmethod
    def _action_asset_kind(lowered_path: str, suffix: str) -> str:
        if suffix in {".gif", ".mp4", ".webm"}:
            return "playback"
        if "contact_sheet" in lowered_path:
            return "contact_sheet"
        if "visual_summary" in lowered_path:
            return "visual_summary"
        if "manifest" in lowered_path:
            return "manifest"
        if "seed_search" in lowered_path:
            return "seed_search"
        if suffix == ".json":
            return "episode_data"
        return "asset"

    @staticmethod
    def _action_asset_label(lowered_path: str, kind: str, map_name: str | None, coverage_hint: float | None) -> str:
        if kind == "playback":
            prefix = "Best" if "best_seed" in lowered_path else "Representative" if "representative" in lowered_path else "Sanity"
            text = f"{prefix} playback"
        elif kind == "contact_sheet":
            text = "Contact sheet"
        elif kind == "visual_summary":
            text = "Visual summary"
        elif kind == "manifest":
            text = "Sanity manifest"
        elif kind == "seed_search":
            text = "Seed search"
        elif kind == "episode_data":
            text = "Episode JSON"
        else:
            text = "Sanity asset"

        parts = [text]
        if map_name:
            parts.append(map_name)
        if coverage_hint is not None:
            parts.append(f"{coverage_hint * 100:.1f}%")
        return " · ".join(parts)

    @staticmethod
    def _action_asset_score(lowered_path: str, suffix: str, map_name: str | None, coverage_hint: float | None) -> int:
        score = {
            ".gif": 34,
            ".mp4": 34,
            ".webm": 34,
            ".png": 16,
            ".json": 11,
            ".md": 6,
            ".csv": 5,
        }.get(suffix, 0)
        if "sanity" in lowered_path:
            score += 32
        if "replay" in lowered_path or "playback" in lowered_path:
            score += 28
        if "best_seed" in lowered_path:
            score += 24
        if "representative" in lowered_path:
            score += 18
        if "contact_sheet" in lowered_path:
            score += 14
        if "visual_summary" in lowered_path:
            score += 10
        if "manifest" in lowered_path:
            score += 8
        if "seed_search" in lowered_path:
            score += 6
        if "sneaky_enemies" in lowered_path:
            score += 4
        if map_name:
            score += 4
        if coverage_hint is not None:
            score += int(round(coverage_hint * 100))
        return score

    def _load_evaluation_points(self, run_dir: Path) -> list[dict[str, Any]]:
        evaluations = _load_json(run_dir / "evaluations.json")
        items = evaluations.get("items", []) if isinstance(evaluations, dict) else []
        points: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            summary = item.get("summary", {})
            timesteps = item.get("timesteps")
            if timesteps is None:
                continue
            points.append(
                {
                    "timesteps": timesteps,
                    "mean_coverage": summary.get("mean_coverage"),
                    "success_rate": summary.get("success_rate"),
                    "death_rate": summary.get("death_rate"),
                    "mean_reward": summary.get("mean_reward"),
                }
            )
        points.sort(key=lambda row: row.get("timesteps") or 0)
        return points

    def _load_map_eval_summaries(self, run_dir: Path) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for map_dir in sorted(path for path in run_dir.glob("map_eval*") if path.is_dir()):
            csv_path = map_dir / "per_map_summary.csv"
            rows = self._read_csv_rows(csv_path)
            if not rows:
                continue
            summaries.append(
                {
                    "map_eval_dir": map_dir.relative_to(self.paths.root).as_posix(),
                    "source_name": map_dir.name,
                    "rows": rows,
                    "episodes_total": sum(int(_safe_float(row.get("episodes")) or 0) for row in rows),
                }
            )
        return summaries

    def _select_primary_map_eval(self, summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not summaries:
            return None

        ranked = sorted(
            summaries,
            key=lambda item: (
                _map_eval_rank(str(item.get("source_name") or "")),
                -(item.get("episodes_total") or 0),
                str(item.get("map_eval_dir") or ""),
            ),
        )
        chosen = ranked[0]
        rows = list(chosen.get("rows") or [])
        if not rows:
            return None

        coverage_values = [_safe_float(row.get("mean_coverage")) for row in rows]
        clean_coverage = [value for value in coverage_values if value is not None]

        return {
            "source_dir": chosen.get("map_eval_dir"),
            "source_name": chosen.get("source_name"),
            "map_count": len(rows),
            "episodes_total": chosen.get("episodes_total") or 0,
            "mean_coverage": _mean_from_rows(rows, "mean_coverage"),
            "mean_success_rate": _mean_from_rows(rows, "success_rate"),
            "mean_death_rate": _mean_from_rows(rows, "death_rate"),
            "mean_reward": _mean_from_rows(rows, "mean_reward"),
            "min_coverage": min(clean_coverage) if clean_coverage else None,
            "max_coverage": max(clean_coverage) if clean_coverage else None,
            "rows": rows,
        }

    def _parse_run_timestamp(self, run_name: str) -> str | None:
        prefix = run_name.split("-", maxsplit=2)
        if len(prefix) < 2:
            return None
        stamp = f"{prefix[0]}-{prefix[1]}"
        try:
            parsed = datetime.strptime(stamp, "%Y%m%d-%H%M%S")
        except ValueError:
            return None
        return parsed.isoformat(timespec="seconds")

    def _normalize_repo_path(self, raw_path: Any) -> str | None:
        if not isinstance(raw_path, str) or raw_path.strip() == "":
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.resolve().relative_to(self.paths.root.resolve()).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def _run_sort_key(run: dict[str, Any]) -> tuple[str, str]:
        timestamp = run.get("timestamp") or ""
        updated_at = run.get("updated_at") or ""
        return timestamp, updated_at
