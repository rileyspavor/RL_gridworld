# Dashboard Backend / Integration Layer

This repo now includes a lightweight local backend for a training dashboard.

It is intentionally simple:
- **no new third-party web framework**
- **reads the repo directly** (`configs/`, `runs/`, `results/`)
- **serves JSON over HTTP** for a frontend/dashboard
- **supports a file-backed queue** for future training launches

The code lives in:
- `rl_coverage/dashboard_data.py` – indexing, summaries, artifact discovery, queue helpers
- `rl_coverage/dashboard_api.py` – local HTTP API + snapshot/queue CLI

---

## What it indexes

### Configs
From `configs/**/*.toml`:
- raw TOML content
- resolved config with defaults applied
- experiment / algorithm / observation / reward / training sections
- launch command template for `python -m rl_coverage.train --config ...`
- linked run ids for completed or partial runs that reference the config

### Runs
From `runs/**` recursively, including nested trees like `runs/finetune/...`:
- `config.json`
- `summary.json`
- `final_evaluation.json`
- `latest_evaluation.json`
- `evaluations.json`
- `best_model_metrics.json`
- `manual_evaluation.json`
- `manual_evaluation_summary.json`
- `map_eval*/per_map_evaluation.json`
- plots / CSVs / model zips / tensorboard files discovered as artifacts

For each run it exposes:
- stable `run_id`
- core labels: algorithm / observation / reward / environment / map suite
- run group (`default`, `finetune`, etc.) based on the relative path under `runs/`
- final metrics summary
- latest metrics summary if present
- best checkpoint from the evaluation curve
- manual evaluation summary if present
- artifact counts by type
- full artifact list on the detail endpoint
- per-map summaries on the detail endpoint

### Results
From `results/**`:
- grouped result collections by directory
- plots, leaderboards, markdown summaries, JSON findings
- lightweight previews for JSON / CSV / markdown files

### Queue
From `dashboard_queue/{pending,running,completed,failed}/`:
- queued job specs as JSON
- command template for future execution

No worker/daemon is included yet. The queue is meant to be a clean local handoff point for a later launcher.

---

## HTTP API

Start the API from repo root:

```bash
python -m rl_coverage.dashboard_api serve --host 127.0.0.1 --port 8765
```

Main endpoints:

- `GET /api/health`
- `GET /api/index`
- `GET /api/leaderboard`
- `GET /api/runs`
- `GET /api/runs/<run_id>`
- `GET /api/configs`
- `GET /api/configs/<config_id>`
- `GET /api/results`
- `GET /api/queue`
- `POST /api/queue`
- `GET /api/command?config_path=configs/...toml&seed=7&output_dir=runs/...`
- `GET /artifacts/<repo-relative-path>`

The `/artifacts/...` route serves files directly from the repo so a frontend can render images and download JSON/CSV/model files without extra glue.

Example:

```text
/artifacts/runs/20260325-192838-ppo_frontier_dense/map_eval/per_map_coverage.png
```

---

## Snapshot mode

If the dashboard does not want a live server yet, generate one static JSON file:

```bash
python -m rl_coverage.dashboard_api snapshot \
  --output results/dashboard_snapshot.json
```

That file contains:
- repo paths
- counts
- leaderboard
- config catalog
- run catalog
- results catalog
- queue state

This is the easiest integration point for a first frontend.

---

## Queue usage

Queue a training job from a known config path:

```bash
python -m rl_coverage.dashboard_api queue-add \
  --config-path configs/experiments/ppo_frontier_dense.toml \
  --seed 11 \
  --output-dir runs/dashboard-ppo-frontier-dense-seed11 \
  --notes "queued from dashboard"
```

Or queue from an indexed config id:

```bash
python -m rl_coverage.dashboard_api queue-add --config-id <config-id>
```

Inspect queue state:

```bash
python -m rl_coverage.dashboard_api queue-list
```

Each queued job stores:
- stable job id
- config id/path
- optional seed/output dir overrides
- notes
- extra args
- canonical training command as both argv and shell string

A future worker can consume the JSON files from `dashboard_queue/pending/` and move them through `running/`, `completed/`, or `failed/`.

---

## Data contract highlights

### `GET /api/runs`
This is the **lightweight run list** for cards/tables.

Each item includes roughly:

```json
{
  "id": "run-...",
  "name": "20260325-192838-ppo_frontier_dense",
  "relative_path": "runs/20260325-192838-ppo_frontier_dense",
  "group": "default",
  "status": "completed",
  "config_id": "cfg-...",
  "config_path": "configs/experiments/ppo_frontier_dense.toml",
  "algorithm": "PPO",
  "observation": "frontier_features",
  "reward": "dense_coverage",
  "environment": "sneaky_enemies",
  "map_suite": "core_generalization",
  "metrics": {
    "mean_coverage": 0.5232,
    "success_rate": 0.25,
    "death_rate": 0.6667,
    "timeout_rate": 0.0833,
    "mean_reward": 32.88,
    "mean_length": 79.92,
    "episodes": 24
  },
  "best_checkpoint": {
    "timesteps": 325000,
    "mean_coverage": 0.5565,
    "success_rate": 0.25
  },
  "artifact_counts": {
    "plot": 4,
    "model": 2,
    "evaluation_history": 1
  },
  "detail_url": "/api/runs/run-..."
}
```

### `GET /api/runs/<run_id>`
This is the **full detail payload**.

It adds:
- full config payload copied from `config.json`
- `summary.json`
- `final_evaluation.json`
- `latest_evaluation.json`
- `best_model_metrics.json`
- manual evaluation payloads if present
- learning curve points from `evaluations.json`
- per-map evaluation summaries from `map_eval*`
- concrete artifact list with `file_url`

### `GET /api/configs`
Useful for a “new experiment” panel.

Each config includes:
- stable config id
- resolved config sections
- launch command template
- linked run ids

### `GET /api/results`
Useful for a report/results browser.

It groups files by directory and includes small previews for:
- JSON findings
- CSV leaderboards
- markdown summaries

---

## Frontend integration pattern

### Option A: live local API
Good when the dashboard needs refreshable run state.

1. Start:
   ```bash
   python -m rl_coverage.dashboard_api serve
   ```
2. Frontend loads:
   - `/api/index` for the initial page
   - `/api/runs/<run_id>` for drill-down
   - `/artifacts/...` for PNG/CSV/JSON/model file access

### Option B: static snapshot
Good when the dashboard is just a local data explorer.

1. Generate:
   ```bash
   python -m rl_coverage.dashboard_api snapshot --output results/dashboard_snapshot.json
   ```
2. Frontend loads that JSON file directly.

---

## Notes about ids

The backend uses stable ids derived from repo-relative paths:
- config ids are based on `configs/...`
- run ids are based on `runs/...`
- result collection ids are based on `results/...`

That keeps ids stable across refreshes as long as paths do not change.

---

## What is intentionally not implemented yet

This backend is a functional integration layer, not a full experiment platform.

Still left for future expansion:
- an actual queue worker that executes pending jobs automatically
- cancellation / retry semantics for queued jobs
- auth/multi-user access controls
- websocket/live streaming during training
- tensorboard scalar extraction instead of just file discovery
- richer result parsing for arbitrary custom report formats
- frontend-specific pagination/filtering if the run count grows large

For now, the important part is in place: the dashboard has a reliable way to discover configs, completed runs, learning curves, artifacts, plots, and queueable training commands from the existing repo layout.
