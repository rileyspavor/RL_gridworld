from __future__ import annotations

import json
import re
import shlex
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


def _slugify(text: str) -> str:
    lowered = text.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "queued-experiment"


def _set_nested(payload: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current = payload
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _trim_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


class ExperimentQueue:
    def __init__(self, queue_file: Path) -> None:
        self.queue_file = queue_file
        self._lock = threading.Lock()

    def list_entries(self) -> list[dict[str, Any]]:
        if not self.queue_file.exists():
            return []

        entries: list[dict[str, Any]] = []
        try:
            with self.queue_file.open("r") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        entries.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []

        entries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return entries

    def build_preview(self, spec: dict[str, Any], config_templates: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = self._normalize_spec(spec)
        template = self._find_template(normalized["base_config"], config_templates)
        if template is None:
            raise ValueError(f"Unknown base config: {normalized['base_config']}")

        overrides: dict[str, Any] = {}
        changed_fields: list[str] = []

        mappings = {
            "experiment_name": (("experiment", "name"), "name"),
            "algorithm": (("algorithm", "name"), "algorithm"),
            "observation": (("observation", "name"), "observation"),
            "reward": (("reward", "name"), "reward"),
            "map_suite": (("environment", "map_suite"), "map_suite"),
            "total_timesteps": (("training", "total_timesteps"), "total_timesteps"),
            "seed": (("experiment", "seed"), "seed"),
        }

        for field, (path_keys, template_key) in mappings.items():
            value = normalized.get(field)
            if value is None or value == "":
                continue
            if template.get(template_key) == value:
                continue
            _set_nested(overrides, path_keys, value)
            changed_fields.append(field)

        requires_materialized = any(
            field in changed_fields
            for field in ("experiment_name", "algorithm", "observation", "reward", "map_suite", "total_timesteps")
        )
        suggested_name = _slugify(_trim_text(normalized.get("experiment_name")) or f"{template.get('name', 'queued')}-variant")
        suggested_config_path = f"configs/experiments/queued/{suggested_name}.toml" if requires_materialized else normalized["base_config"]

        command_parts = ["python", "-m", "rl_coverage.train", "--config", suggested_config_path]
        if normalized.get("seed") is not None:
            command_parts += ["--seed", str(normalized["seed"])]
        if normalized.get("output_dir"):
            command_parts += ["--output-dir", normalized["output_dir"]]

        command_preview = " ".join(shlex.quote(part) for part in command_parts)

        return {
            "normalized_spec": normalized,
            "base_template": template,
            "changed_fields": changed_fields,
            "config_overrides": overrides,
            "requires_materialized_config": requires_materialized,
            "suggested_config_path": suggested_config_path,
            "command_preview": command_preview,
        }

    def enqueue(self, spec: dict[str, Any], config_templates: list[dict[str, Any]]) -> dict[str, Any]:
        preview = self.build_preview(spec, config_templates)
        normalized = preview["normalized_spec"]

        created_at = datetime.now().isoformat(timespec="seconds")
        slug_source = _trim_text(normalized.get("experiment_name")) or _trim_text(normalized.get("base_config"))
        entry_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_slugify(slug_source)}"

        entry = {
            "id": entry_id,
            "created_at": created_at,
            "status": "queued",
            "base_config": normalized["base_config"],
            "notes": normalized.get("notes", ""),
            "spec": normalized,
            "preview": {
                "command_preview": preview["command_preview"],
                "changed_fields": preview["changed_fields"],
                "config_overrides": preview["config_overrides"],
                "requires_materialized_config": preview["requires_materialized_config"],
                "suggested_config_path": preview["suggested_config_path"],
            },
        }

        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.queue_file.open("a") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def _find_template(self, config_path: str, templates: list[dict[str, Any]]) -> dict[str, Any] | None:
        for template in templates:
            if template.get("path") == config_path:
                return template
        return None

    def _normalize_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "base_config": _trim_text(spec.get("base_config")),
            "experiment_name": _trim_text(spec.get("experiment_name")),
            "algorithm": _trim_text(spec.get("algorithm")),
            "observation": _trim_text(spec.get("observation")),
            "reward": _trim_text(spec.get("reward")),
            "map_suite": _trim_text(spec.get("map_suite")),
            "output_dir": _trim_text(spec.get("output_dir")),
            "notes": _trim_text(spec.get("notes")),
            "seed": self._parse_int(spec.get("seed")),
            "total_timesteps": self._parse_int(spec.get("total_timesteps")),
        }
        if not normalized["base_config"]:
            raise ValueError("'base_config' is required")
        return normalized

    def _parse_int(self, value: Any) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"Expected integer value, got '{value}'") from exc
