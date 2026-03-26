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

## Recommended setups

If you only try a few experiments, start here:

1. **Best overall starting point:** `ppo_frontier_dense.toml`
2. **Safer / more conservative alternative:** `ppo_frontier_survival.toml`
3. **Full-grid structured alternative:** `ppo_layered_dense.toml`
4. **Baseline to beat:** `dqn_rgb_baseline.toml`

### Why `ppo_frontier_dense` is the current best bet

Without even needing a giant hyperparameter sweep, the environment itself gives away a few things:

- The task is **long horizon**.
- The agent needs **coverage planning**, not just reflexive obstacle avoidance.
- The raw RGB observation is unnecessarily hard for an MLP.
- Sparse rewards make credit assignment much worse.
- PPO tends to be more forgiving than DQN on shaped continuous-ish feature inputs in tasks like this.

So the strongest practical default is:

- **Algorithm:** PPO
- **Observation:** `frontier_features`
- **Reward:** `dense_coverage`
- **Training maps:** `core_generalization`

That combination gives the agent direct access to the information it actually needs:
frontier direction, local move quality, coverage progress, and danger structure.

### What I would expect to happen empirically

- `ppo_frontier_dense` should learn fastest.
- `ppo_frontier_survival` should reduce catastrophic deaths, possibly at the cost of slightly slower coverage.
- `ppo_layered_dense` may catch up if you train longer because it keeps more spatial detail.
- `dqn_rgb_baseline` is mostly there as a reference point and should usually be weaker and slower.

If you have time to run only one serious training job, run `ppo_frontier_dense` first.

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
