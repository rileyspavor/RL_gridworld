# Results Integration Guide (Submission Snapshot)

Use this file as the canonical map between report claims and repository artifacts.

---

## 1) Primary evidence bundles

### Controlled 3×3 PPO observation/reward matrix
- `results/observation_reward_sweep/leaderboard.csv`
- `results/observation_reward_sweep/summary.md`
- `results/observation_reward_sweep/learning_curves.png`
- `results/observation_reward_sweep/mean_coverage_by_combo.png`
- `results/observation_reward_sweep/mean_reward_by_combo.png`
- `results/observation_reward_sweep/success_rate_by_combo.png`

### Long-run frontier baseline
- `runs/20260325-192838-ppo_frontier_dense/summary.json`
- `runs/20260325-192838-ppo_frontier_dense/best_model_metrics.json`
- `runs/20260325-192838-ppo_frontier_dense/final_evaluation.json`

### Strategic continuation finalists
- `runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish/manual_evaluation_summary.json`
- `runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish/map_eval_128_baseline/per_map_summary.csv`
- `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_all_standard_lowlr/manual_evaluation_summary.json`
- `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_all_standard_lowlr/map_eval_128_final/per_map_summary.csv`

---

## 2) Claim discipline used in final report

- "Best in controlled matrix" refers only to the 3×3 PPO ablation.
- "Best broad mixed-suite score" refers to 128-episode `all_standard` manual evaluation.
- "Best hard-map-balanced checkpoint" refers to per-map average and `chokepoint`/`sneaky_enemies` behavior, not only global mean.
- Any cross-phase comparison is explicitly labeled and not treated as direct apples-to-apples unless evaluation protocol matches.

---

## 3) Canonical report table source

`report/RESULTS_TABLE_TEMPLATE.csv` is now populated from the matrix leaderboard and should be treated as the canonical table source for observation/reward comparison claims.

---

## 4) Rebuild commands (if numbers/plots need refresh)

```bash
python -m rl_coverage.plot_runs --runs-dir runs/sweep --output-dir results/observation_reward_sweep
python -m rl_coverage.compare --runs-dir runs/sweep > results/sweep_compare.txt
```

Per-map reevaluation examples:

```bash
python -m rl_coverage.eval_maps --run-dir runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish --model best_model --episodes 128 --output-dir map_eval_128_baseline
python -m rl_coverage.eval_maps --run-dir runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_all_standard_lowlr --model best_model --episodes 128 --output-dir map_eval_128_final
```

---

## 5) Current known gaps

- Multi-seed confidence intervals are not included.
- DQN is not re-run in the newest strategic continuation phase.
- `runs/` artifacts are intentionally ignored in git and must be archived separately for full reproducibility outside this machine.
