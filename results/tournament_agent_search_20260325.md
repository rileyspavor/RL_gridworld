# Tournament agent search — 2026-03-25

## Winner

**Model:** `runs/finetune/20260325-195113-ppo_frontier_dense_hard_finetune/best_model.zip`

**Training strategy:**
1. Start from the existing best general PPO model: `runs/20260325-192838-ppo_frontier_dense/final_model.zip`
2. Fine-tune on `hard_only` (`chokepoint`, `sneaky_enemies`) with the same `frontier_features + dense_coverage` setup
3. Select the best checkpoint by evaluation mean coverage during fine-tuning

## Best aggregate metrics found

### Full standard map set (`all_standard`, 60 deterministic episodes)
- **Winner hard-only fine-tune best checkpoint:** coverage **0.656**, success 0.200, death 0.517, timeout 0.283, mean length 183.5
- Baseline long-run final model: coverage 0.585, success 0.200, death 0.550, timeout 0.250, mean length 156.0
- Improvement over baseline: **+0.071 mean coverage**

### Core tournament/generalization suite (`core_generalization`, 60 deterministic episodes)
- **Winner hard-only fine-tune best checkpoint:** coverage **0.568**, success 0.000, death 0.633, timeout 0.367, mean length 199.2
- Baseline long-run final model: coverage 0.513, success 0.250, death 0.700, timeout 0.050, mean length 64.3
- Improvement over baseline: **+0.055 mean coverage**

## Winner per-map performance (`32` deterministic episodes per map)
Artifacts: `runs/finetune/20260325-195113-ppo_frontier_dense_hard_finetune/map_eval_best_model_32eps/`

- `just_go`: coverage **1.000**, success 1.000, death 0.000
- `safe`: coverage **0.971**, success 0.000, death 0.000
- `maze`: coverage **0.744**, success 0.000, death 0.656
- `chokepoint`: coverage **0.369**, success 0.000, death 0.969
- `sneaky_enemies`: coverage **0.182**, success 0.000, death 1.000

## Other experiments tried

### 1) Hard-only fine-tune + mixed-suite polish
Run: `runs/finetune/20260325-195251-ppo_frontier_dense_hard_then_core_polish`
- Helped recover some completion behavior on easy maps
- `all_standard` 60-episode coverage: **0.608**
- Worse than the plain hard-only fine-tune winner

### 2) Hard-only survival-leaning fine-tune
Run: `runs/finetune/20260325-195402-ppo_frontier_survival_hard_finetune`
- `all_standard` 60-episode coverage: **0.574**
- Worse than both baseline and dense hard-only fine-tune on coverage

### 3) Second hard-only continuation from the winning hard-only checkpoint
Run: `runs/finetune/20260325-195525-ppo_frontier_dense_hard_finetune_stage2`
- Overfit badly to hard maps
- `all_standard` 60-episode coverage: **0.390**
- Hard-map coverage rose, but easy-map coverage collapsed; rejected

## Code changes made

- `rl_coverage/train.py`
  - Added optional checkpoint initialization via `algorithm.init_model_path`
  - Added support for `training.reset_num_timesteps` during resumed/fine-tuned training
- Added configs:
  - `configs/experiments/ppo_frontier_dense_hard_finetune.toml`
  - `configs/experiments/ppo_frontier_survival_hard_finetune.toml`
  - `configs/experiments/ppo_frontier_dense_hard_then_core_polish.toml`
  - `configs/experiments/ppo_frontier_dense_hard_finetune_stage2.toml`
  - `configs/experiments/ppo_frontier_dense_all_standard_eval.toml`

## Recurrent / LSTM PPO feasibility

- `stable_baselines3` is installed
- `sb3_contrib` is **not** installed in the local `.venv`
- Therefore **RecurrentPPO/LSTM PPO was not feasible in the current stack without installing new dependencies**
- I did **not** install anything new; I stayed within the existing environment

## Hard constraint check

- `coverage-gridworld/coverage_gridworld/env.py` was **not modified**
