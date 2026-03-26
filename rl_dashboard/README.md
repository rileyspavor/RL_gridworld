# RL Dashboard

Lightweight local dashboard for this repo’s existing RL artifacts. It is built with:

- Python stdlib HTTP server + JSON API
- repo-native JSON/TOML/CSV parsing (`tomllib`, `csv`, `json`)
- vanilla HTML/CSS/JS frontend

No extra runtime installs are required.

## What it surfaces

- completed and in-progress runs discovered from `runs/**`
- run-level summaries from `summary.json`, `final_evaluation.json`, and `evaluations.json`
- comparison tables for algorithm/observation/reward combinations
- embedded plot/image artifacts from:
  - `runs/**/map_eval*/`
  - `results/observation_reward_sweep/*.png`
- leaderboard and report docs from `results/**`
- `report/EXPERIMENT_INBOX.md` preview
- queue-ready experiment planner that writes specs to `report/EXPERIMENT_QUEUE.jsonl`

## Architecture

- `rl_dashboard/models.py`
  - shared path model (`DashboardPaths`)
- `rl_dashboard/data.py`
  - run/config/results discovery + artifact loading
- `rl_dashboard/summaries.py`
  - filtering and aggregate comparison summaries
- `rl_dashboard/queue_store.py`
  - experiment queue persistence + command/config preview generation
- `rl_dashboard/server.py`
  - lightweight HTTP server, API routes, static/artifact serving
- `rl_dashboard/static/*`
  - browser UI

## Run

From repo root:

```bash
python -m rl_dashboard --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

Optional args:

- `--root <path>`: repository root to scan
- `--queue-file <path>`: queue file path (default: `report/EXPERIMENT_QUEUE.jsonl`)

## Queue behavior

Planner entries are appended as JSONL records to:

```text
report/EXPERIMENT_QUEUE.jsonl
```

Each queued record includes:

- normalized form inputs
- generated training command preview
- config override preview (for non-CLI-overridable fields)
- suggested materialized config path for future queued variants

This keeps experiment planning config-driven and easy to hand off to future run orchestration.
