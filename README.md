# Coverage Gridworld RL Project

This repository contains a cleaned-up reinforcement learning project for the Coverage Gridworld assignment.

The project is built around the assignment goals:

- use **one RL algorithm**: PPO
- implement **at least two observation spaces**
- implement **at least three reward functions**
- compare them experimentally
- train the best agent you can
- generate plots and write up the results

The final code keeps `env.py` untouched and performs all customization through `custom.py`, feature engineering, reward shaping, and training configuration.

Tournament submittion package `Project Code/tournament_zips/` 

If you already have a working environment, the minimum commands are:

```bash
pip install -r requirements-train.txt
pip install -e coverage-gridworld
```

```bash
python play.py --run-dir runs/matrix/comp_model_80_play --sleep 0.1
```

---

# 1. What this repository is

This repo has **two related but distinct parts**.

## A. The clean assignment pipeline

This is the main project and the one that should be treated as the official assignment implementation.

It contains:

- PPO training
- the 2 observation spaces used in the assignment matrix
- the 3 reward functions used in the assignment matrix
- scripts to train, evaluate, compare, and render agents
- plots and summaries for the report

## B. Compatibility support for `comp_model_80`

The repo also contains a compatibility path for a recovered checkpoint named `comp_model_80`.

That model does **not** use the same observation size as the clean assignment matrix. The assignment matrix uses compact custom feature vectors, while `comp_model_80` expects a richer **142-dimensional** observation vector.

So the codebase supports both:

- a **clean assignment experiment framework**, and
- a **compatibility mode** so the recovered checkpoint can still be loaded and played.

---

# 2. High-level architecture

The easiest way to understand the repo is to think of it in layers.

## Layer 1: The environment package

Location:

- `coverage-gridworld/coverage_gridworld/`

Important files:

- `env.py` — the original environment implementation
- `custom.py` — the assignment-approved customization entrypoint

### What this layer does

This is the actual game/environment.
It defines:

- the maps
- the enemies
- the action loop
- episode termination
- the raw grid observation

### Important rule

`env.py` is intentionally **not modified**.

That matters because the assignment explicitly said environment changes should not be made there.

---

## Layer 2: Runtime customization bridge

Main file:

- `coverage-gridworld/coverage_gridworld/custom.py`

Supporting file:

- `project_rl/customization.py`

### What this layer does

This layer is the bridge between the course environment and the custom RL logic.

The environment calls into `custom.py` for:

- observation space
- observation transformation
- reward computation

Instead of hardcoding everything in `custom.py`, the file delegates to configurable implementations in `project_rl`.

### Why this is useful

It keeps the assignment hook simple while still allowing:

- multiple observation-space variants
- multiple reward-function variants
- config-driven switching between experiments

---

## Layer 3: RL logic and experiment internals

Location:

- `project_rl/`

This is where the main project logic lives.

### Main modules

- `observations.py`
  - implements feature-based observation spaces
- `rewards.py`
  - implements reward functions
- `grid_utils.py`
  - low-level grid parsing and geometry helpers
- `maps.py`
  - standard map definitions and map suites
- `env_factory.py`
  - builds environments from configs
- `training.py`
  - PPO training orchestration
- `metrics.py`
  - evaluation summaries and aggregate metrics
- `callbacks.py`
  - periodic evaluation during training
- `plotting.py`
  - CSV and plot generation
- `config.py`
  - TOML/JSON config loading and run-artifact saving

### What this layer does

This layer turns the assignment from a single custom file into a structured RL project.

It handles:

- feature engineering
- reward shaping
- experiment configuration
- environment setup
- PPO training
- evaluation and plots

---

## Layer 4: User-facing scripts

Files at repo root:

- `train.py`
- `evaluate.py`
- `play.py`
- `run_experiments.py`

### What this layer does

These are the command-line entrypoints you actually run.

- `train.py`
  - train a single config
- `evaluate.py`
  - evaluate a saved model
- `play.py`
  - render a model playing the environment
- `run_experiments.py`
  - run the full assignment experiment matrix and generate report plots

These scripts are intentionally kept at the repo root so they are easy to discover and easy to run.

---

## Layer 5: Configs, outputs, and report artifacts

Important folders/files:

- `configs/`
- `runs/`
- `results/plots/`
- `tournament_zips/`

### What this layer does

This layer stores:

- experiment definitions
- trained run artifacts
- evaluation summaries
- plots used in the report
- the final tournament submittion

---

# 3. Directory guide

Here is the practical layout of the repo.

```text
Project Code/
├── README.md
├── tournament_zips/
│   └── comp_model_80.zip/ 
│   └── custom.py.zip/ 
├── requirements-train.txt
├── train.py
├── evaluate.py
├── play.py
├── run_experiments.py
├── configs/
│   ├── experiments/
│   └── best_agent/
├── results/
│   └── plots/
├── runs/
│   └── matrix/
├── comp_model_80/
├── comp_model_80.zip
├── project_rl/
│   ├── __init__.py
│   ├── bootstrap.py
│   ├── callbacks.py
│   ├── config.py
│   ├── customization.py
│   ├── env_factory.py
│   ├── grid_utils.py
│   ├── maps.py
│   ├── metrics.py
│   ├── observations.py
│   ├── plotting.py
│   ├── rewards.py
│   └── training.py
└── coverage-gridworld/
    ├── setup.py
    └── coverage_gridworld/
        ├── __init__.py
        ├── env.py
        └── custom.py
```

---

# 4. Observation spaces used in this project

## 1. `frontier_features`

This is the compact observation used in the clean assignment matrix.

It summarizes things like:

- agent position
- coverage progress
- nearby cell categories
- directional freedom of movement
- frontier distance cues

### Why it exists

It gives PPO a small, task-relevant state description instead of making it learn directly from raw RGB pixels.

---

## 2. `temporal_frontier_features`

This is the richer clean assignment observation.

It includes the frontier features above, plus temporal danger information such as:

- future enemy field-of-view masks
- safe-action information under future enemy rotations
- compact enemy orientation/location summaries

### Why it exists

The environment is not just about coverage. Timing matters because enemies rotate every step. This observation helps the policy reason about near-future danger.

---

## 3. `strategic_temporal_frontier_features`

This is the compatibility observation for `comp_model_80`.

It is **not part of the clean assignment matrix**.

It exists so the recovered checkpoint can load correctly. It includes richer handcrafted features such as:

- temporal safety features
- reachable-region/action metrics
- map identity information

### Why it exists

Without this compatibility path, `comp_model_80` cannot be evaluated or rendered correctly because its network expects a different observation vector size.

---

# 5. Reward functions used in this project

## 1. `sparse_coverage`

A relatively simple reward focused on:

- step penalty
- reward for covering a new cell
- terminal bonus/penalty

### Purpose

Acts as the simplest baseline.

---

## 2. `dense_coverage`

Adds more shaping around:

- coverage ratio
- progress speed
- new coverage
- terminal outcomes

### Purpose

Provides smoother learning feedback than sparse reward.

---

## 3. `survival_coverage`

Adds safety-sensitive shaping such as:

- enemy pressure / observed-cell penalties
- progress terms
- survival-oriented terminal structure

### Purpose

Better reflects the actual stealth-and-coverage objective of the environment.

---

# 6. Assignment experiment design

The clean assignment experiment matrix is:

- **1 algorithm:** PPO
- **2 observation spaces:**
  - `frontier_features`
  - `temporal_frontier_features`
- **3 reward functions:**
  - `sparse_coverage`
  - `dense_coverage`
  - `survival_coverage`

That gives the required **2 x 3** comparison matrix.

Config files live in:

- `configs/experiments/`

These six configs are the clean assignment runs:

- `frontier_features_sparse_coverage.toml`
- `frontier_features_dense_coverage.toml`
- `frontier_features_survival_coverage.toml`
- `temporal_frontier_features_sparse_coverage.toml`
- `temporal_frontier_features_dense_coverage.toml`
- `temporal_frontier_features_survival_coverage.toml`

---

# 7. Best-agent training path

After the 2x3 matrix is complete, the best configuration can be extended with a longer training path.

Configs for that live in:

- `configs/best_agent/`

These are intended for the assignment goal of training the best agent possible after comparing observation/reward choices.

---

# 8. How the data flows through the project

This is the important mental model.

## During training

1. `train.py` loads a TOML config.
2. `project_rl/config.py` merges it with defaults.
3. `project_rl/env_factory.py` builds environments using that config.
4. `coverage_gridworld.custom` is configured with the selected observation/reward variants.
5. PPO trains using those observation and reward definitions.
6. `project_rl/callbacks.py` periodically evaluates the model.
7. `project_rl/metrics.py` and `project_rl/config.py` save summaries.

## During evaluation

1. `evaluate.py` loads a run config and checkpoint.
2. The environment is rebuilt the same way as during training.
3. The checkpoint is run for a chosen number of episodes.
4. Aggregate metrics are printed and saved.

## During playback

1. `play.py` loads a config and checkpoint.
2. The environment is created in human render mode.
3. The trained policy acts in real time so behavior can be inspected visually.

---

# 9. Setup

From the repo root, create or activate a Python environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
pip install -e coverage-gridworld
```

If you already have a working environment, the minimum commands are:

```bash
pip install -r requirements-train.txt
pip install -e coverage-gridworld
```

---

# 10. Common commands

## Train one config

```bash
python train.py --config configs/experiments/frontier_features_dense_coverage.toml
```

## Train one config with overrides

```bash
python train.py --config configs/experiments/frontier_features_dense_coverage.toml --timesteps 50000 --seed 11
```

## Evaluate a saved run

```bash
python evaluate.py --run-dir runs/matrix/<run-folder>
```

## Evaluate from explicit config + model

```bash
python evaluate.py --config configs/experiments/frontier_features_dense_coverage.toml --model runs/matrix/<run-folder>/best_model
```

## Watch a trained model play

By default, `play.py` plays one episode per map in the configured suite.

For `all_standard`, that means **5 episodes** by default.

```bash
python play.py --run-dir runs/matrix/<run-folder> --sleep 0.1
```

## Force a specific number of episodes

```bash
python play.py --run-dir runs/matrix/<run-folder> --episodes 3 --sleep 0.1
```

## Run the full 2x3 matrix

```bash
python run_experiments.py
```

## Run the matrix and then continue the best model

```bash
python run_experiments.py --train-best --best-timesteps 350000
```

---

# 11. Important results and artifacts

## Plots

Location:

- `results/plots/`

Important generated files:

- `leaderboard.csv`
- `mean_coverage_by_combo.png`
- `success_rate_by_combo.png`
- `mean_reward_by_combo.png`
- `learning_curves.png`
- `coverage_heatmap.png`

## Run manifest

Location:

- `runs/matrix/latest_matrix_runs.json`

This file summarizes the clean assignment matrix results.

---

# 12. Best clean assignment result

From the finished 2x3 matrix, the best clean assignment setup was:

- **Observation:** `temporal_frontier_features`
- **Reward:** `survival_coverage`

This was the strongest result among the clean assignment runs and is the main result the report should focus on.

---

# 13. `comp_model_80` usage

`comp_model_80` is kept in the repo as a recovered checkpoint that now works with the compatibility observation path.

## Evaluate it

```bash
python evaluate.py --run-dir runs/matrix/comp_model_80_play
```

## Watch it play

```bash
python play.py --run-dir runs/matrix/comp_model_80_play --sleep 0.1
```

## Important note

This model is **not the clean assignment matrix winner**.  
It is a separate recovered model supported by compatibility code.

---

# 14. What to look at first if you are grading or reviewing the code

If someone wants the shortest path to understanding the project, these are the most important files:

- `README.md`
- `coverage-gridworld/coverage_gridworld/custom.py`
- `project_rl/observations.py`
- `project_rl/rewards.py`
- `project_rl/training.py`
- `train.py`
- `evaluate.py`
- `play.py`
- `run_experiments.py`
- `configs/experiments/`
- `results/plots/`

---

# 15. Practical summary

If you want the shortest possible explanation of the repo:

- the **environment** lives in `coverage-gridworld/coverage_gridworld/`
- `custom.py` is the official assignment hook
- the actual RL logic lives in `project_rl/`
- the root scripts are the commands you run
- `configs/` defines experiments
- `runs/` stores run artifacts
- `results/plots/` stores report figures
- `comp_model_80` is supported as a separate compatibility story

That is the architecture in one page.

Authors: Riley Spavor, Brandon Liang, Rowan Mohammed, George Salib, Karina Verma