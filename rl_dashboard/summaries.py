from __future__ import annotations

from statistics import fmean
from typing import Any


STANDARD_MAP_ORDER = ("just_go", "safe", "maze", "chokepoint", "sneaky_enemies")
DEFAULT_RUN_SORT = "coverage"


def sort_runs(runs: list[dict[str, Any]], sort_by: str = "", order: str = "") -> list[dict[str, Any]]:
    normalized_sort = _normalize_run_sort(sort_by)
    ascending = str(order or "").strip().lower() == "asc"

    if normalized_sort == "family":
        sorted_runs = sorted(
            runs,
            key=lambda run: (
                str(run.get("family") or "").lower(),
                -(_metric_value(run, "mean_coverage") or 0.0),
                str(run.get("run_name") or ""),
            ),
        )
        if not ascending:
            sorted_runs.reverse()
        return sorted_runs

    def numeric_value(run: dict[str, Any]) -> float | None:
        if normalized_sort == "coverage":
            return _metric_value(run, "mean_coverage")
        if normalized_sort == "map_mean":
            return _primary_map_value(run, "mean_coverage")
        if normalized_sort == "timesteps":
            return _float_value(run.get("total_timesteps"))
        if normalized_sort == "success":
            return _metric_value(run, "success_rate")
        return _map_metric(run, normalized_sort, "mean_coverage")

    def sort_key(run: dict[str, Any]) -> tuple[int, float, float, str]:
        value = numeric_value(run)
        coverage = _metric_value(run, "mean_coverage") or 0.0
        run_name = str(run.get("run_name") or "")
        if value is None:
            return (1, 0.0, -coverage, run_name)
        adjusted = value if ascending else -value
        return (0, adjusted, -coverage, run_name)

    return sorted(runs, key=sort_key)


def completed_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run for run in runs if run.get("status") == "completed"]


def filter_runs(runs: list[dict[str, Any]], filters: dict[str, str]) -> list[dict[str, Any]]:
    filtered = runs
    for field in ("family", "status", "algorithm", "observation", "reward", "map_suite"):
        value = filters.get(field, "").strip()
        if not value:
            continue
        filtered = [run for run in filtered if str(run.get(field, "")).lower() == value.lower()]

    search_text = filters.get("q", "").strip().lower()
    if search_text:
        filtered = [
            run
            for run in filtered
            if search_text in str(run.get("run_name", "")).lower()
            or search_text in str(run.get("relative_dir", "")).lower()
            or search_text in str(run.get("config_path", "")).lower()
            or search_text in str(run.get("family", "")).lower()
        ]
    return filtered


def available_options(runs: list[dict[str, Any]], configs: list[dict[str, Any]]) -> dict[str, list[str]]:
    fields = ("family", "algorithm", "observation", "reward", "map_suite")
    options: dict[str, set[str]] = {field: set() for field in fields}

    for run in runs:
        for field in fields:
            value = run.get(field)
            if value:
                options[field].add(str(value))

    for config in configs:
        for field in ("algorithm", "observation", "reward", "map_suite"):
            value = config.get(field)
            if value:
                options[field].add(str(value))

    return {field: sorted(values) for field, values in options.items()}


def summarize_combinations(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in completed_runs(runs):
        key = (
            str(run.get("algorithm") or "unknown"),
            str(run.get("observation") or "unknown"),
            str(run.get("reward") or "unknown"),
        )
        buckets.setdefault(key, []).append(run)

    combinations: list[dict[str, Any]] = []
    for (algorithm, observation, reward), bucket in buckets.items():
        metrics = [item.get("metrics", {}) for item in bucket]
        mean_coverage = _mean_metric(metrics, "mean_coverage")
        success_rate = _mean_metric(metrics, "success_rate")
        mean_reward = _mean_metric(metrics, "mean_reward")
        death_rate = _mean_metric(metrics, "death_rate")
        timeout_rate = _mean_metric(metrics, "timeout_rate")
        timesteps = _mean_values(item.get("total_timesteps") for item in bucket)
        combinations.append(
            {
                "algorithm": algorithm,
                "observation": observation,
                "reward": reward,
                "run_count": len(bucket),
                "mean_coverage": mean_coverage,
                "success_rate": success_rate,
                "mean_reward": mean_reward,
                "death_rate": death_rate,
                "timeout_rate": timeout_rate,
                "avg_total_timesteps": timesteps,
                "run_ids": [item["run_id"] for item in bucket],
            }
        )

    combinations.sort(key=lambda row: ((row.get("mean_coverage") or 0.0), (row.get("success_rate") or 0.0)), reverse=True)
    return combinations


def summarize_overview(
    runs: list[dict[str, Any]],
    combinations: list[dict[str, Any]],
    queue_entries: list[dict[str, Any]],
    configs: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = completed_runs(runs)
    ranked = sorted(
        completed,
        key=lambda row: (
            row.get("metrics", {}).get("mean_coverage") or 0.0,
            row.get("metrics", {}).get("success_rate") or 0.0,
        ),
        reverse=True,
    )
    best_run = ranked[0] if ranked else None
    latest_completed = sorted(completed, key=lambda row: row.get("updated_at") or "", reverse=True)[0] if completed else None

    statuses: dict[str, int] = {}
    for run in runs:
        key = str(run.get("status") or "unknown")
        statuses[key] = statuses.get(key, 0) + 1

    baseline_reference = _select_baseline_reference(completed)
    balanced_candidate = _select_balanced_candidate(completed)
    sneaky_specialist = _select_sneaky_specialist(completed)
    current_best_model = _select_current_best_model(completed)
    pure_coverage_winner = current_best_model
    map_comparison = _build_map_comparison(
        completed,
        balanced_candidate=balanced_candidate,
        pure_coverage_winner=pure_coverage_winner,
        sneaky_specialist=sneaky_specialist,
        baseline_reference=baseline_reference,
    )

    return {
        "counts": {
            "runs_total": len(runs),
            "runs_completed": len(completed),
            "configs_total": len(configs),
            "combinations_total": len(combinations),
            "queue_total": len(queue_entries),
        },
        "status_breakdown": statuses,
        "best_run": _trim_run(best_run),
        "latest_completed": _trim_run(latest_completed),
        "top_runs": [_trim_run(item) for item in ranked[:5]],
        "featured": {
            "current_best_model": _highlight_run(
                current_best_model,
                role="Current best",
                score_name="overall_coverage",
                score_value=_metric_value(current_best_model, "mean_coverage"),
                explanation="Highest overall coverage among high-signal completed runs.",
            ),
            "balanced_agent": _highlight_run(
                balanced_candidate,
                role="Balanced winner",
                score_name="balanced_score",
                score_value=_balanced_score(balanced_candidate),
                explanation="Best cross-map balance using standard-map mean, floor, and success.",
            ),
            "sneaky_specialist": _highlight_run(
                sneaky_specialist,
                role="Sneaky specialist",
                score_name="sneaky_coverage",
                score_value=_map_metric(sneaky_specialist, "sneaky_enemies", "mean_coverage"),
                explanation="Highest sneaky_enemies coverage in the preferred per-map evaluation.",
            ),
            "pure_coverage_winner": _highlight_run(
                pure_coverage_winner,
                role="Pure coverage winner",
                score_name="overall_coverage",
                score_value=_metric_value(pure_coverage_winner, "mean_coverage"),
                explanation="Highest aggregate run coverage from the main run summary.",
            ),
            "baseline_reference": _highlight_run(
                baseline_reference,
                role="Baseline reference",
                score_name="map_mean_coverage",
                score_value=_primary_map_value(baseline_reference, "mean_coverage"),
                explanation="Best root-level baseline run to compare experiments against.",
            ),
            "best_combo": combinations[0] if combinations else None,
        },
        "map_comparison": map_comparison,
        "map_leaders": _build_map_leaders(completed),
        "experiment_deltas": _build_experiment_deltas(completed, baseline_reference),
    }



def _highlight_run(
    run: dict[str, Any] | None,
    *,
    role: str,
    score_name: str,
    score_value: float | None,
    explanation: str,
) -> dict[str, Any] | None:
    if not run:
        return None
    payload = _trim_run(run)
    payload.update(
        {
            "role": role,
            "score_name": score_name,
            "score_value": score_value,
            "explanation": explanation,
            "sneaky_coverage": _map_metric(run, "sneaky_enemies", "mean_coverage"),
            "chokepoint_coverage": _map_metric(run, "chokepoint", "mean_coverage"),
            "maze_coverage": _map_metric(run, "maze", "mean_coverage"),
        }
    )
    return payload


def _trim_run(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if not run:
        return None
    return {
        "run_id": run.get("run_id"),
        "run_name": run.get("run_name"),
        "family": run.get("family"),
        "algorithm": run.get("algorithm"),
        "observation": run.get("observation"),
        "reward": run.get("reward"),
        "map_suite": run.get("map_suite"),
        "timestamp": run.get("timestamp"),
        "updated_at": run.get("updated_at"),
        "metrics": run.get("metrics", {}),
        "total_timesteps": run.get("total_timesteps"),
        "primary_map_eval": run.get("primary_map_eval"),
        "has_action_assets": bool(run.get("has_action_assets")),
        "top_action_asset": run.get("top_action_asset"),
    }


def _mean_metric(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in metrics if item.get(key) is not None]
    if not values:
        return None
    return float(fmean(values))


def _mean_values(values: Any) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(fmean(clean))


def _metric_value(run: dict[str, Any] | None, key: str) -> float | None:
    if not run:
        return None
    metrics = run.get("metrics", {}) or {}
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _primary_map_value(run: dict[str, Any] | None, key: str) -> float | None:
    if not run:
        return None
    primary = run.get("primary_map_eval") or {}
    value = primary.get(key)
    if value is None:
        return None
    return float(value)


def _map_rows(run: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not run:
        return []
    primary = run.get("primary_map_eval") or {}
    rows = primary.get("rows") or []
    return [row for row in rows if isinstance(row, dict)]


def _map_row(run: dict[str, Any] | None, map_name: str) -> dict[str, Any] | None:
    for row in _map_rows(run):
        if str(row.get("map_name") or "") == map_name:
            return row
    return None


def _map_metric(run: dict[str, Any] | None, map_name: str, key: str) -> float | None:
    row = _map_row(run, map_name)
    if not row:
        return None
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def _balanced_score(run: dict[str, Any] | None) -> float | None:
    rows = _map_rows(run)
    if len(rows) < 4:
        return None
    coverage_values = [float(row["mean_coverage"]) for row in rows if row.get("mean_coverage") is not None]
    success_values = [float(row["success_rate"]) for row in rows if row.get("success_rate") is not None]
    death_values = [float(row["death_rate"]) for row in rows if row.get("death_rate") is not None]
    if not coverage_values:
        return None
    mean_cov = float(fmean(coverage_values))
    min_cov = min(coverage_values)
    mean_success = float(fmean(success_values)) if success_values else 0.0
    mean_death = float(fmean(death_values)) if death_values else 0.0
    return mean_cov + (0.25 * mean_success) + (0.15 * min_cov) - (0.05 * mean_death)


def _select_balanced_candidate(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [run for run in runs if not _is_low_signal(run) and _balanced_score(run) is not None]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda run: (
            _balanced_score(run) or 0.0,
            _primary_map_value(run, "mean_coverage") or 0.0,
            _metric_value(run, "success_rate") or 0.0,
        ),
        reverse=True,
    )[0]


def _select_sneaky_specialist(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [run for run in runs if not _is_low_signal(run) and _map_metric(run, "sneaky_enemies", "mean_coverage") is not None]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda run: (
            _map_metric(run, "sneaky_enemies", "mean_coverage") or 0.0,
            _primary_map_value(run, "mean_coverage") or 0.0,
            _metric_value(run, "mean_reward") or 0.0,
        ),
        reverse=True,
    )[0]


def _select_current_best_model(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [run for run in runs if not _is_low_signal(run)]
    pool = candidates or runs
    if not pool:
        return None
    return sorted(
        pool,
        key=lambda run: (
            _metric_value(run, "mean_coverage") or 0.0,
            _primary_map_value(run, "mean_coverage") or 0.0,
            _map_metric(run, "sneaky_enemies", "mean_coverage") or 0.0,
            _metric_value(run, "success_rate") or 0.0,
        ),
        reverse=True,
    )[0]


def _select_baseline_reference(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        run
        for run in runs
        if str(run.get("family") or "") == "baseline"
        and not _is_low_signal(run)
        and _primary_map_value(run, "mean_coverage") is not None
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda run: (
            _primary_map_value(run, "mean_coverage") or 0.0,
            _metric_value(run, "mean_coverage") or 0.0,
            _metric_value(run, "success_rate") or 0.0,
        ),
        reverse=True,
    )[0]


def _comparison_entry(role: str, run: dict[str, Any]) -> dict[str, Any]:
    rows = _map_rows(run)
    map_scores = {str(row.get("map_name")): row for row in rows if row.get("map_name")}
    return {
        "role": role,
        **(_trim_run(run) or {}),
        "balanced_score": _balanced_score(run),
        "map_mean_coverage": _primary_map_value(run, "mean_coverage"),
        "sneaky_coverage": _map_metric(run, "sneaky_enemies", "mean_coverage"),
        "map_scores": map_scores,
    }


def _build_map_comparison(
    runs: list[dict[str, Any]],
    *,
    balanced_candidate: dict[str, Any] | None,
    pure_coverage_winner: dict[str, Any] | None,
    sneaky_specialist: dict[str, Any] | None,
    baseline_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    selected: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    def push(role: str, run: dict[str, Any] | None) -> None:
        if not run:
            return
        run_id = str(run.get("run_id") or "")
        if not run_id or run_id in seen:
            return
        selected.append((role, run))
        seen.add(run_id)

    push("Current best", pure_coverage_winner)
    push("Balanced winner", balanced_candidate)
    push("Sneaky specialist", sneaky_specialist)
    push("Baseline", baseline_reference)

    contenders = sorted(
        [run for run in runs if not _is_low_signal(run) and _balanced_score(run) is not None and str(run.get("run_id") or "") not in seen],
        key=lambda run: (_balanced_score(run) or 0.0, _primary_map_value(run, "mean_coverage") or 0.0),
        reverse=True,
    )
    while len(selected) < 4 and contenders:
        contender = contenders.pop(0)
        push("Contender", contender)

    map_order = [map_name for map_name in STANDARD_MAP_ORDER if any(_map_row(run, map_name) for _, run in selected)]
    return {
        "maps": map_order,
        "items": [_comparison_entry(role, run) for role, run in selected],
    }


def _build_map_leaders(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaders: list[dict[str, Any]] = []
    candidates = [run for run in runs if not _is_low_signal(run)]
    for map_name in STANDARD_MAP_ORDER:
        ranked = sorted(
            [run for run in candidates if _map_metric(run, map_name, "mean_coverage") is not None],
            key=lambda run: (
                _map_metric(run, map_name, "mean_coverage") or 0.0,
                _map_metric(run, map_name, "success_rate") or 0.0,
                -(_map_metric(run, map_name, "death_rate") or 0.0),
            ),
            reverse=True,
        )
        if not ranked:
            continue
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        winner_cov = _map_metric(winner, map_name, "mean_coverage") or 0.0
        runner_cov = _map_metric(runner_up, map_name, "mean_coverage") if runner_up else None
        leaders.append(
            {
                "map_name": map_name,
                "winner": {
                    **(_trim_run(winner) or {}),
                    "coverage": winner_cov,
                    "success_rate": _map_metric(winner, map_name, "success_rate"),
                    "death_rate": _map_metric(winner, map_name, "death_rate"),
                },
                "runner_up": (
                    {
                        **(_trim_run(runner_up) or {}),
                        "coverage": runner_cov,
                    }
                    if runner_up
                    else None
                ),
                "gap": (winner_cov - runner_cov) if runner_cov is not None else None,
            }
        )
    return leaders


def _build_experiment_deltas(runs: list[dict[str, Any]], baseline_reference: dict[str, Any] | None) -> dict[str, Any]:
    if not baseline_reference:
        return {"reference": None, "improvements": [], "regressions": []}

    baseline_map_mean = _primary_map_value(baseline_reference, "mean_coverage")
    baseline_sneaky = _map_metric(baseline_reference, "sneaky_enemies", "mean_coverage")
    if baseline_map_mean is None:
        return {"reference": _trim_run(baseline_reference), "improvements": [], "regressions": []}

    changes: list[dict[str, Any]] = []
    for run in runs:
        if run.get("run_id") == baseline_reference.get("run_id"):
            continue
        if _is_low_signal(run) or str(run.get("family") or "") == "sweep":
            continue
        map_mean = _primary_map_value(run, "mean_coverage")
        if map_mean is None:
            continue
        sneaky = _map_metric(run, "sneaky_enemies", "mean_coverage")
        changes.append(
            {
                **(_trim_run(run) or {}),
                "delta_map_mean_coverage": map_mean - baseline_map_mean,
                "delta_sneaky_coverage": (sneaky - baseline_sneaky) if sneaky is not None and baseline_sneaky is not None else None,
            }
        )

    improvements = sorted(
        [item for item in changes if (item.get("delta_map_mean_coverage") or 0.0) > 0.01],
        key=lambda item: (item.get("delta_map_mean_coverage") or 0.0, item.get("delta_sneaky_coverage") or 0.0),
        reverse=True,
    )[:4]
    regressions = sorted(
        [item for item in changes if (item.get("delta_map_mean_coverage") or 0.0) < -0.01],
        key=lambda item: (item.get("delta_map_mean_coverage") or 0.0, item.get("delta_sneaky_coverage") or 0.0),
    )[:4]

    return {
        "reference": _trim_run(baseline_reference),
        "improvements": improvements,
        "regressions": regressions,
    }


def _is_low_signal(run: dict[str, Any]) -> bool:
    run_name = str(run.get("run_name") or "").lower()
    family = str(run.get("family") or "")
    if "smoke" in run_name:
        return True
    if family == "sweep":
        return True
    if len(_map_rows(run)) < 4:
        return True
    return False


def _normalize_run_sort(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if text.startswith("map:"):
        map_name = text.split(":", maxsplit=1)[1].strip()
        if map_name in STANDARD_MAP_ORDER:
            return map_name

    aliases = {
        "": DEFAULT_RUN_SORT,
        "coverage": "coverage",
        "overall": "coverage",
        "overall_coverage": "coverage",
        "map_mean": "map_mean",
        "mean_map": "map_mean",
        "map": "map_mean",
        "timesteps": "timesteps",
        "steps": "timesteps",
        "family": "family",
        "success": "success",
        "success_rate": "success",
        "sneaky": "sneaky_enemies",
        "sneaky_enemies": "sneaky_enemies",
        "chokepoint": "chokepoint",
        "just_go": "just_go",
        "safe": "safe",
        "maze": "maze",
    }
    return aliases.get(text, DEFAULT_RUN_SORT)
