# RL Coverage Project Findings (Submission Snapshot)

## Executive summary
- Controlled 3×3 PPO ablation confirms `frontier_features` is the strongest observation family in early fixed-budget training.
- Long-run `ppo_frontier_dense` substantially outperforms short sweeps but leaves hard-map gaps.
- Strategic temporal continuation runs produce the highest broad coverage seen in this repository state.
- The top broad scalar score and top hard-map-balanced checkpoint are close but not identical.

---

## 1) Controlled 3×3 PPO ablation (seed 21, `core_generalization`)

Budget/run: 50k steps, 4 envs, eval every 10k, 8 eval episodes.

| Rank | Observation | Reward | Coverage | Success | Death | Mean reward | Run |
|---|---|---|---:|---:|---:|---:|---|
| 1 | frontier_features | dense_coverage | 0.458 | 0.000 | 0.667 | 5.354 | `20260325-192639-ppo_frontier_dense_sweep` |
| 2 | frontier_features | survival_coverage | 0.352 | 0.000 | 0.667 | -9.042 | `20260325-192702-ppo_frontier_survival_sweep` |
| 3 | frontier_features | sparse_coverage | 0.316 | 0.000 | 0.750 | 14.217 | `20260325-192651-ppo_frontier_sparse_sweep` |
| 4 | layered_grid | survival_coverage | 0.243 | 0.000 | 0.417 | -74.405 | `20260325-192740-ppo_layered_survival_sweep` |
| 5 | layered_grid | dense_coverage | 0.240 | 0.000 | 0.500 | -47.624 | `20260325-192713-ppo_layered_dense_sweep` |
| 6 | raw_rgb | dense_coverage | 0.149 | 0.000 | 0.500 | -53.933 | `20260325-192752-ppo_rgb_dense_sweep` |
| 7 | layered_grid | sparse_coverage | 0.147 | 0.000 | 0.000 | 2.417 | `20260325-192726-ppo_layered_sparse_sweep` |
| 8 | raw_rgb | survival_coverage | 0.137 | 0.000 | 0.333 | -95.737 | `20260325-192810-ppo_rgb_survival_sweep` |
| 9 | raw_rgb | sparse_coverage | 0.121 | 0.000 | 0.500 | 1.709 | `20260325-192800-ppo_rgb_sparse_sweep` |

Primary read: representation quality mattered more than reward choice in this budget regime.

---

## 2) Long-run frontier baseline

Run: `runs/20260325-192838-ppo_frontier_dense`

- Final deterministic eval (24 episodes):
  - mean coverage **0.523**
  - success **0.250**
  - death **0.667**
  - timeout **0.083**
- Best checkpoint during training (~325k, 16 episodes): mean coverage **0.557**

This run is a strong mid-project anchor and a reproducible baseline for later continuation phases.

---

## 3) Strategic temporal continuation phase

### Strongest broad mixed-suite score
- Run: `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_all_standard_lowlr`
- 128-episode manual broad eval: mean coverage **0.847791**, success **0.515625**, death **0.187500**, timeout **0.296875**

### Best hard-map-balanced checkpoint
- Run: `runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish`
- 128-episode manual broad eval: mean coverage **0.846203**, success **0.562500**, death **0.195313**, timeout **0.242188**
- Per-map 128-episode means:
  - `just_go`: **1.000000**
  - `safe`: **1.000000**
  - `maze`: **0.995763**
  - `chokepoint`: **0.665249**
  - `sneaky_enemies`: **0.519271**

Comparison note: low-LR broad continuation wins by scalar mean coverage; hard-emphasis safe-polish retains stronger `chokepoint` and per-map balance.

---

## 4) Submission recommendation

If selecting one final model for tournament-style robustness, use:

- `runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish/best_model.zip`

If selecting purely by broad mixed-suite mean coverage scalar, use:

- `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_all_standard_lowlr/best_model.zip`

---

## 5) Artifact pointers

- Sweep artifacts: `results/observation_reward_sweep/`
- Long baseline run: `runs/20260325-192838-ppo_frontier_dense/`
- Strategic run notes:
  - `results/strategic_recovery_search_20260326.md`
  - `results/safe_polish_continuation_search_20260326.md`
  - `results/tournament_agent_search_20260325.md`
