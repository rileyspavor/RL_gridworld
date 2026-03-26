# Coverage Gridworld RL Project

![visualization](media/sneaky_enemies.gif "Sneaky Enemies sample layout")

This repo now has two layers:

1. **`coverage-gridworld/`** – the environment package.
2. **`rl_coverage/`** – training, evaluation, comparison, config, and experiment tooling.

The goal is still the same: learn policies that maximize safe map coverage as quickly as possible.
The difference is that observation spaces, reward functions, map suites, and algorithms are now cleanly swappable.

---

## What was here originally

The original repo had:

- a custom Gymnasium environment,
- a manual-play script (`main.py`),
- a very small DQN training example (`model.py`),
- a placeholder `custom.py` where observation/reward design was expected to happen.

That was enough to start, but not enough to compare experiments cleanly or iterate quickly.

---

## What the project looks like now

```text
Project Code/
├── configs/
│   └── experiments/
│       ├── dqn_rgb_baseline.toml
│       ├── ppo_frontier_dense.toml
│       ├── ppo_frontier_survival.toml
│       └── ppo_layered_dense.toml
├── coverage-gridworld/
│   └── coverage_gridworld/
│       ├── __init__.py
│       ├── custom.py
│       └── env.py
├── rl_coverage/
│   ├── bootstrap.py
│   ├── callbacks.py
│   ├── compare.py
│   ├── config.py
│   ├── env_factory.py
│   ├── evaluate.py
│   ├── grid_utils.py
│   ├── maps.py
│   ├── metrics.py
│   ├── observations.py
│   ├── play.py
│   ├── rewards.py
│   └── train.py
├── main.py
├── model.py
└── requirements-train.txt
```

---

## Environment summary

### Rules

The agent starts at `(0, 0)` and must cover all reachable cells while avoiding enemy line-of-sight.
Enemies rotate counter-clockwise each step and their field of view is blocked by walls and other enemies.

### Standard maps

These are exposed both as Gym ids and as reusable map data:

- `just_go`
- `safe`
- `maze`
- `chokepoint`
- `sneaky_enemies`

### Map suites

The experiment layer adds named suites for easy multi-map training:

- `all_standard`
- `easy_to_hard`
- `hard_only`
- `core_generalization`
- `tournament_hard_emphasis`
- `tournament_safe_polish`
- `tournament_hard_anchor`
- `tournament_sneaky_anchor`

---

## Observation variants

Observation swapping is handled outside the environment through wrappers.
That means you can compare representations without rewriting env internals.

### `raw_rgb`
- Normalized flattened RGB grid.
- Closest to the original project setup.
- Good baseline, but inefficient because the model must learn color semantics from scratch.

### `layered_grid`
- Binary channels for unexplored, observed-unexplored, explored, observed-explored, walls, enemies, and agent.
- Keeps the full board structure.
- Usually a better default than raw RGB if you want global spatial information.

### `frontier_features`
- Compact handcrafted feature vector.
- Includes agent position, coverage ratios, local move affordances, local neighborhood categories, directional frontier/danger signals, nearest frontier offsets, quadrant frontier density, and reachable frontier distance.
- This is the most targeted representation for MLP policies and is the recommended starting point.

### `temporal_frontier_features`
- Extends `frontier_features` with short-horizon danger forecasts and action-conditioned safety/readiness signals.
- Better for policies that must avoid delayed enemy line-of-sight traps.

### `strategic_temporal_frontier_features`
- Extends `temporal_frontier_features` with reachable-component metrics and map identity features.
- This is the strongest representation family in the latest tournament-style runs.

---

## Reward variants

Reward swapping is also handled with wrappers, so the same environment can be trained with different shaping logic.

### `sparse_coverage`
- Simple baseline.
- Small step penalty, reward for new coverage, penalty for death/timeouts, completion bonus.

### `dense_coverage`
- Recommended starting reward.
- Adds progress shaping, revisit penalties, invalid-move penalties, stay penalties, and loop discouragement.
- Good balance between exploration pressure and optimization stability.

### `survival_coverage`
- More conservative than `dense_coverage`.
- Stronger death/loop/stall penalties.
- Best when policies keep dying and need stronger pressure toward safe trajectories.

---

## Submission Snapshot (measured)

The repository now has completed quantitative evidence beyond the initial design hypothesis.

### Core observation/reward ablation (PPO, 3×3, seed 21)
- Best of the 9-way sweep: `frontier_features + dense_coverage` with mean coverage **0.458** on `core_generalization` at 50k steps.
- Worst of the 9-way sweep: `raw_rgb + sparse_coverage` with mean coverage **0.121**.
- Full table and plots: `results/observation_reward_sweep/leaderboard.csv` and `results/observation_reward_sweep/summary.md`.
- Full report-quality write-up: `report/REPORT_DRAFT.md` and `results/README.md`.

### Strongest broad-coverage continuation
- `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_all_standard_lowlr/manual_evaluation_summary.json`
- 128 deterministic episodes on `all_standard`: mean coverage **0.848**, success **0.516**, death **0.188**.

### Best hard-map-balanced checkpoint
- `runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish/best_model.zip`
- 128-episode per-map eval shows stronger `chokepoint` coverage (**0.665**) than the low-LR broad continuation (**0.650**) while keeping easy maps near-solved.

### Recommended configs by objective
1. **Best broad mixed-suite score:** `configs/experiments/ppo_strategic_temporal_safe_polish_all_standard_lowlr.toml`
2. **Best hard-map balance:** `configs/experiments/ppo_strategic_temporal_hard_emphasis_safe_polish.toml`
3. **Best simple baseline to reproduce quickly:** `configs/experiments/ppo_frontier_dense.toml`
4. **Reward/observation starter benchmark:** `configs/experiments/sweep/ppo_frontier_dense_sweep.toml`

---

## Installation

Create a local environment and install the training dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
```

You do **not** need a separate editable install step just to run the project from this repo root, because `rl_coverage/bootstrap.py` adds the local environment package path automatically.

If you want the environment installed independently for outside use, you can still do:

```bash
pip install -e coverage-gridworld
```

---

## How to train

### Recommended run

```bash
python -m rl_coverage.train --config configs/experiments/ppo_frontier_dense.toml
```

This is the safest cold-start config. Strategic tournament configs in `configs/experiments/ppo_strategic_*.toml` are continuation-style and expect existing checkpoint paths.

### Other built-in experiments

```bash
python -m rl_coverage.train --config configs/experiments/ppo_frontier_survival.toml
python -m rl_coverage.train --config configs/experiments/ppo_layered_dense.toml
python -m rl_coverage.train --config configs/experiments/dqn_rgb_baseline.toml
```

### Override the output directory

```bash
python -m rl_coverage.train \
  --config configs/experiments/ppo_frontier_dense.toml \
  --output-dir runs/manual-ppo-frontier-dense
```

### Override the seed

```bash
python -m rl_coverage.train \
  --config configs/experiments/ppo_frontier_dense.toml \
  --seed 123
```

---

## How evaluation works

During training, periodic evaluation writes:

- `latest_evaluation.json`
- `evaluations.json`
- `best_model.zip` (if enabled)
- `best_model_metrics.json`

At the end of training, the run directory also gets:

- `final_model.zip`
- `final_evaluation.json`
- `summary.json`
- `config.json`

### Evaluate an existing run

```bash
python -m rl_coverage.evaluate --run-dir runs/<your-run-dir>
```

### Evaluate a specific model path

```bash
python -m rl_coverage.evaluate \
  --config configs/experiments/ppo_frontier_dense.toml \
  --model runs/<your-run-dir>/best_model
```

---

## How to compare runs

Once you have multiple completed experiments:

```bash
python -m rl_coverage.compare --runs-dir runs
```

That prints a compact leaderboard using the saved `summary.json` files.

---

## Dashboard backend / integration layer

There is now a lightweight local backend for dashboard/frontend work.
It reads `configs/`, `runs/`, and `results/` directly and exposes them as structured JSON.

### Start the local API

```bash
python -m rl_coverage.dashboard_api serve --host 127.0.0.1 --port 8765
```

### Write a static snapshot instead

```bash
python -m rl_coverage.dashboard_api snapshot --output results/dashboard_snapshot.json
```

### Queue a future training job

```bash
python -m rl_coverage.dashboard_api queue-add \
  --config-path configs/experiments/ppo_frontier_dense.toml \
  --seed 11 \
  --output-dir runs/dashboard-ppo-frontier-dense-seed11
```

See `docs/dashboard_backend.md` for the API/data contract and frontend integration notes.

---

## Manual play / visual inspection

### Human play

```bash
python main.py --policy human --env-id sneaky_enemies
```

### Random play

```bash
python main.py --policy random --env-id sneaky_enemies --sleep 0.1
```

### Play using an experiment config

```bash
python main.py --config configs/experiments/ppo_frontier_dense.toml --policy human
```

---

## Notes on the environment package

A couple of cleanups were made in `coverage-gridworld/`:

- `custom.py` now exposes a sensible default flattened `Box` observation space instead of a `MultiDiscrete` RGB encoding.
- `custom.py` now returns a non-zero default reward so quick tests are meaningful.
- `setup.py` now includes `numpy`, which the environment already depends on.
- `__init__.py` now exposes reusable map constants and map-cloning helpers instead of hiding the map data only inside registration calls.

---

## Suggested next experiments

If you want to push performance further, I’d do these next:

1. **Longer PPO training** for `ppo_frontier_dense` and `ppo_layered_dense`
2. **Curriculum vs hard-only maps** comparison
3. **Reward weight sweeps** around revisit / death penalties
4. **Policy architecture sweeps** (larger MLPs, maybe attention or CNN if you move to 2D observations)
5. **Fine-tuning on `sneaky_enemies` after curriculum pretraining**

---

## Quick project diagnosis

### Original strengths
- The environment itself is solid and interesting.
- The map design has meaningful difficulty tiers.
- The task has real structure, so representation design matters.

### Original weaknesses
- No experiment system
- No clean config layer
- No reusable observation/reward variants
- The provided DQN example was too thin to support serious comparison

### Current state
- Clean experiment entry points
- Reusable map suites
- Swappable observation wrappers
- Swappable reward wrappers
- Better defaults
- Saved metrics and comparable run artifacts

That should make the repo much easier to train, tune, and extend.
