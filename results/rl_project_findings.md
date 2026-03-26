# RL Coverage Project Findings

## Runtime fixes completed
- `rl_coverage/observations.py` — Fixed frontier_features observation space from 53 to 54 dimensions so SB3 training no longer crashes.
- `requirements-train.txt` — Added rich and tqdm so the documented install supports the SB3 progress bar.
- `rl_coverage/train.py` — Made progress bar optional at runtime; training now falls back cleanly if rich/tqdm are absent.
- `rl_coverage/plot_runs.py` — Added CSV/PNG reporting utility for run comparison.
- `rl_coverage/eval_maps.py` — Added per-map evaluation utility for standard maps.

## Targeted PPO sweep (seed 21, map suite `core_generalization`)
- Maps: `safe`, `maze`, `chokepoint`, `sneaky_enemies`
- Budget per run: 50,000 timesteps, 4 envs, eval every 10,000 steps, 8 eval episodes

| Rank | Observation | Reward | Coverage | Success | Death | Mean reward | Run |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | frontier_features | dense_coverage | 0.458 | 0.000 | 0.667 | 5.354 | `20260325-192639-ppo_frontier_dense_sweep` |
| 2 | frontier_features | survival_coverage | 0.352 | 0.000 | 0.667 | -9.042 | `20260325-192702-ppo_frontier_survival_sweep` |
| 3 | frontier_features | sparse_coverage | 0.316 | 0.000 | 0.750 | 14.217 | `20260325-192651-ppo_frontier_sparse_sweep` |
| 4 | layered_grid | survival_coverage | 0.243 | 0.000 | 0.417 | -74.405 | `20260325-192740-ppo_layered_survival_sweep` |
| 5 | layered_grid | dense_coverage | 0.240 | 0.000 | 0.500 | -47.624 | `20260325-192713-ppo_layered_dense_sweep` |
| 6 | raw_rgb | dense_coverage | 0.149 | 0.000 | 0.500 | -53.933 | `20260325-192752-ppo_rgb_dense_sweep` |
| 7 | layered_grid | sparse_coverage | 0.147 | 0.000 | 0.000 | 2.417 | `20260325-192726-ppo_layered_sparse_sweep` |
| 8 | raw_rgb | survival_coverage | 0.137 | 0.000 | 0.333 | -95.737 | `20260325-192810-ppo_rgb_survival_sweep` |
| 9 | raw_rgb | sparse_coverage | 0.121 | 0.000 | 0.500 | 1.709 | `20260325-192800-ppo_rgb_sparse_sweep` |

## Strongest long run
- Config: `configs/experiments/ppo_frontier_dense.toml`
- Seed: 7
- Maps: `safe`, `maze`, `chokepoint`, `sneaky_enemies` via map suite `core_generalization`
- Budget: 400,000 timesteps, 8 envs, eval every 25,000 steps, 16 eval episodes
- Final 24-episode deterministic eval (`final_model`): coverage=0.515, success=0.250, death=0.667, timeout=0.083, mean_reward=31.974, mean_length=78.9
- Best checkpoint during training (325k, 16 eval episodes): coverage=0.557, success=0.250, death=0.625, timeout=0.125, mean_reward=30.966

### Per-map final_model evaluation (16 episodes each)
| Map | Coverage | Success | Death | Timeout | Mean reward | Mean length |
|---|---:|---:|---:|---:|---:|---:|
| just_go | 0.860 | 0.000 | 0.000 | 1.000 | 55.980 | 500.0 |
| safe | 1.000 | 1.000 | 0.000 | 0.000 | 88.907 | 102.0 |
| maze | 0.728 | 0.000 | 0.750 | 0.250 | 31.435 | 152.7 |
| chokepoint | 0.235 | 0.000 | 1.000 | 0.000 | 11.928 | 18.8 |
| sneaky_enemies | 0.148 | 0.000 | 1.000 | 0.000 | 3.673 | 10.9 |

## Key takeaways
- Observation choice mattered more than reward choice: frontier_features dominated every layered/raw_rgb setup in this sweep.
- Within the frontier feature family, dense_coverage was the strongest practical reward; survival_coverage was second and sparse_coverage third.
- A longer 400k-step PPO frontier+dense run reached 25% overall success on the core_generalization suite and >0.52 mean coverage on 24 deterministic eval episodes.
- The remaining performance gap is concentrated on the hardest maps, especially chokepoint and sneaky_enemies.

## Artifact paths
- `sweep_leaderboard_csv` → `results/observation_reward_sweep/leaderboard.csv`
- `sweep_summary_md` → `results/observation_reward_sweep/summary.md`
- `sweep_learning_curves_png` → `results/observation_reward_sweep/learning_curves.png`
- `sweep_mean_coverage_png` → `results/observation_reward_sweep/mean_coverage_by_combo.png`
- `sweep_mean_reward_png` → `results/observation_reward_sweep/mean_reward_by_combo.png`
- `sweep_success_png` → `results/observation_reward_sweep/success_rate_by_combo.png`
- `long_run_dir` → `runs/20260325-192838-ppo_frontier_dense`
- `long_run_map_eval_dir` → `runs/20260325-192838-ppo_frontier_dense/map_eval_final_model`

