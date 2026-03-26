# Coverage Tournament Report

## Title
**Coverage Tournament: Reinforcement Learning for Safe Exploration in Coverage Gridworld**

## Authors
Riley Spavor et al.  
Queen's University

---

## Abstract
This project targets safe long-horizon coverage in a 10×10 gridworld with rotating enemies and line-of-sight elimination. The core constraint was to keep `coverage-gridworld/coverage_gridworld/env.py` unchanged and improve performance through modular observation/reward wrappers and training strategy. We ran a controlled 3×3 PPO ablation over observation and reward variants, then scaled into longer training and continuation curricula. In the controlled matrix, `frontier_features + dense_coverage` was best (mean coverage **0.458**) and `raw_rgb + sparse_coverage` was worst (**0.121**). The strongest broad mixed-suite score came from a strategic continuation (`strategic_temporal_frontier_features + dense_coverage`) at **0.848** mean coverage over 128 deterministic `all_standard` episodes, while the best hard-map-balanced checkpoint achieved a stronger `chokepoint` result (**0.665** vs **0.650**) with nearly identical broad coverage.

---

## 1. Problem Setup and Constraints
The agent starts at `(0, 0)` and must cover all reachable cells while avoiding enemy vision. Enemies rotate counter-clockwise each step, vision is occlusion-aware, and episodes end on detection, full coverage, or the step budget.

This is difficult because:
1. the objective is long horizon (maximize eventual coverage, not just immediate safety),
2. danger is directional and time-varying,
3. reward credit assignment is sensitive to loops, hesitation, and delayed traps.

Hard project constraint: `coverage-gridworld/coverage_gridworld/env.py` is not modified. All major improvements are implemented in wrappers and training infrastructure.

---

## 2. Method

### 2.1 Algorithms
- Primary algorithm: **PPO** (`rl_coverage/train.py`).
- Also supported: **DQN** and **A2C** (used as baselines/tooling, not the strongest final approach).

### 2.2 Observation Variants
Implemented in `rl_coverage/observations.py`:
- `raw_rgb` (300 features): flattened normalized RGB.
- `layered_grid` (705 features): semantic channels + scalar context.
- `frontier_features` (54 features): compact handcrafted frontier/safety state.
- `temporal_frontier_features` (122 features): frontier features + short-horizon temporal danger/action structure.
- `strategic_temporal_frontier_features` (142 features): temporal features + reachable-component and map-identity features.

### 2.3 Reward Variants
Implemented in `rl_coverage/rewards.py`:
- `sparse_coverage`
- `dense_coverage`
- `survival_coverage`

All are wrapper-based and run on the same environment dynamics.

### 2.4 Training/Evaluation Infrastructure
- Config-driven experiments in `configs/experiments/*.toml`.
- Periodic evaluation, best-checkpoint saving, and final summaries in run artifacts.
- Continuation support with `algorithm.init_model_path` and `training.reset_num_timesteps`.
- Continuation override fix in `rl_coverage/train.py` reapplies low-LR PPO settings after model load.

---

## 3. Experimental Plan Executed

### 3.1 Controlled PPO 3×3 Matrix (Observation × Reward)
- Suite: `core_generalization`
- Maps: `safe`, `maze`, `chokepoint`, `sneaky_enemies`
- Budget/run: 50k steps, 4 envs, eval every 10k, 8 eval episodes (seed 21)

### 3.2 Long-Run Baseline Push
- Config: `configs/experiments/ppo_frontier_dense.toml`
- Budget: 400k steps, 8 envs, eval every 25k, final deterministic eval over 24 episodes

### 3.3 Tournament-Style Continuation Search
Temporal/strategic feature family continuation runs targeted broad `all_standard` performance while preserving hard-map behavior, especially `chokepoint` and `sneaky_enemies`.

---

## 4. Results

### 4.1 PPO 3×3 Observation/Reward Ablation

| Rank | Observation | Reward | Mean coverage | Success | Death | Timeout | Mean reward |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | frontier_features | dense_coverage | 0.458 | 0.000 | 0.667 | 0.333 | 5.354 |
| 2 | frontier_features | survival_coverage | 0.352 | 0.000 | 0.667 | 0.333 | -9.042 |
| 3 | frontier_features | sparse_coverage | 0.316 | 0.000 | 0.750 | 0.250 | 14.217 |
| 4 | layered_grid | survival_coverage | 0.243 | 0.000 | 0.417 | 0.583 | -74.405 |
| 5 | layered_grid | dense_coverage | 0.240 | 0.000 | 0.500 | 0.500 | -47.624 |
| 6 | raw_rgb | dense_coverage | 0.149 | 0.000 | 0.500 | 0.500 | -53.933 |
| 7 | layered_grid | sparse_coverage | 0.147 | 0.000 | 0.000 | 1.000 | 2.417 |
| 8 | raw_rgb | survival_coverage | 0.137 | 0.000 | 0.333 | 0.667 | -95.737 |
| 9 | raw_rgb | sparse_coverage | 0.121 | 0.000 | 0.500 | 0.500 | 1.709 |

Takeaway: observation quality dominates early performance; `frontier_features` consistently beats layered/raw options in this fixed-budget matrix.

### 4.2 Stronger Runs Beyond the Matrix

| Run | Evaluation set | Episodes | Mean coverage | Success | Death | Timeout |
|---|---|---:|---:|---:|---:|---:|
| `20260325-192838-ppo_frontier_dense` final model | `core_generalization` | 24 | 0.523 | 0.250 | 0.667 | 0.083 |
| `...ppo_frontier_dense` best checkpoint (~325k) | periodic eval | 16 | 0.557 | 0.250 | 0.625 | 0.125 |
| `20260326-143040-...hard_emphasis_safe_polish` best model | `all_standard` manual eval | 128 | 0.846 | 0.563 | 0.195 | 0.242 |
| `20260326-144942-...safe_polish_all_standard_lowlr` best model | `all_standard` manual eval | 128 | 0.848 | 0.516 | 0.188 | 0.297 |

### 4.3 Hard-Map Balance Check (Per-Map, 128 Episodes)

| Map | Hard-emphasis safe-polish | All-standard low-LR continuation |
|---|---:|---:|
| `just_go` | 1.000 | 1.000 |
| `safe` | 1.000 | 1.000 |
| `maze` | 0.996 | 0.996 |
| `chokepoint` | **0.665** | 0.650 |
| `sneaky_enemies` | 0.519 | **0.520** |
| Per-map average | **0.836** | 0.833 |

Interpretation: the low-LR continuation slightly improves broad aggregate mean coverage, but the hard-emphasis safe-polish checkpoint remains better on `chokepoint` and per-map hard-balance average.

---

## 5. Best and Worst Approaches

### Best (selected for tournament balance)
**`ppo_strategic_temporal_hard_emphasis_safe_polish`** with `strategic_temporal_frontier_features + dense_coverage`.

Why it wins for the final submission objective:
- near-identical broad mixed-suite coverage to the top scalar score,
- better `chokepoint` coverage,
- strongest per-map average balance across the five standard maps.

### Worst (controlled matrix)
**`raw_rgb + sparse_coverage`** in the PPO sweep (mean coverage **0.121**).

Likely cause: weak intermediate credit assignment plus raw representation burden makes long-horizon safe exploration significantly harder.

---

## 6. Limitations
- Most runs are single-seed; robustness across seeds remains open.
- Evaluation suites differ across phases (controlled matrix vs tournament curricula), so only like-for-like comparisons are treated as decisive.
- Policies are MLP-based on flattened/feature vectors; no CNN/recurrent policy comparison is included in this submission.
- DQN baseline config exists but is not rerun in the latest strategic continuation phase.

---

## 7. Conclusion
The project meets the assignment goals by improving performance through observation design, reward shaping, and training strategy without modifying the base environment dynamics. Evidence now supports a clear progression: compact frontier features outperform raw/layered alternatives in controlled early budgets, and strategic temporal continuations produce much higher broad coverage in tournament-style settings. The final submission recommendation is the hard-emphasis safe-polish strategic checkpoint for best hard-map-balanced performance, with the low-LR all-standard continuation retained as the strongest broad scalar alternative.

---

## Appendix A: Evidence Files
- `results/observation_reward_sweep/leaderboard.csv`
- `results/observation_reward_sweep/summary.md`
- `runs/20260325-192838-ppo_frontier_dense/summary.json`
- `runs/20260325-192838-ppo_frontier_dense/best_model_metrics.json`
- `runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish/manual_evaluation_summary.json`
- `runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish/map_eval_128_baseline/per_map_summary.csv`
- `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_all_standard_lowlr/manual_evaluation_summary.json`
- `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_all_standard_lowlr/map_eval_128_final/per_map_summary.csv`
