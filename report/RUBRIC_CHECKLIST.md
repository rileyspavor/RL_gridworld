# Coverage Tournament Report — Rubric Checklist (Submission State)

This checklist reflects the current repository state and completed evidence.

---

## Assignment constraints from brief
- Do **not** modify `coverage-gridworld/coverage_gridworld/env.py`.
- Use reward shaping and observation-space design to improve performance.
- Use a suitable RL algorithm.
- Provide comparative experiments and plots.
- Provide a comprehensive report with best/worst discussion.

Constraint status: ✅ `env.py` remains untouched.

---

## Rubric evidence

### 1) Implement a suitable RL algorithm `[5]`
**Status:** ✅ Complete

**Evidence:**
- `rl_coverage/train.py` supports PPO, DQN, and A2C.
- Main submission runs use PPO with controlled sweeps and continuation strategies.

### 2) Implement at least two distinct observation spaces `[6]`
**Status:** ✅ Complete (exceeds minimum)

**Evidence (`rl_coverage/observations.py`):**
1. `raw_rgb` (300)
2. `layered_grid` (705)
3. `frontier_features` (54)
4. `temporal_frontier_features` (122)
5. `strategic_temporal_frontier_features` (142)

### 3) Implement at least three unique reward functions `[9]`
**Status:** ✅ Complete

**Evidence (`rl_coverage/rewards.py`):**
1. `sparse_coverage`
2. `dense_coverage`
3. `survival_coverage`

### 4) Experiment with each observation/reward combination and generate plots `[5]`
**Status:** ✅ Complete for PPO matrix

**Evidence:**
- Full 3×3 PPO matrix completed (`raw_rgb`, `layered_grid`, `frontier_features` × `sparse`, `dense`, `survival`).
- Metrics table: `results/observation_reward_sweep/leaderboard.csv`
- Summary: `results/observation_reward_sweep/summary.md`
- Plots:
  - `results/observation_reward_sweep/learning_curves.png`
  - `results/observation_reward_sweep/mean_coverage_by_combo.png`
  - `results/observation_reward_sweep/mean_reward_by_combo.png`
  - `results/observation_reward_sweep/success_rate_by_combo.png`

### 5) Train the best possible agent and use unique strategy if helpful `[5]`
**Status:** ✅ Complete

**Evidence:**
- Long-run baseline: `runs/20260325-192838-ppo_frontier_dense/summary.json`
- Strategic continuation search and finalists:
  - `runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish/`
  - `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_all_standard_lowlr/`
- Broad and per-map evaluations:
  - `.../manual_evaluation_summary.json`
  - `.../map_eval_128_baseline/per_map_summary.csv`
  - `.../map_eval_128_final/per_map_summary.csv`

### 6) Comprehensive report `[10]`
**Status:** ✅ Complete (current draft now evidence-backed)

**Evidence:**
- Updated report with measured best/worst claims: `report/REPORT_DRAFT.md`
- Updated results table with actual matrix metrics: `report/RESULTS_TABLE_TEMPLATE.csv`
- Updated experiment log: `report/EXPERIMENT_INBOX.md`

---

## Final readiness checks
- [x] Constraint respected (`env.py` unchanged)
- [x] Observation and reward implementations documented
- [x] Controlled comparison included
- [x] Plots and leaderboard included
- [x] Best and worst approaches identified with numbers
- [x] Limitations explicitly acknowledged

---

## Remaining caution points (non-blocking)
- Most comparisons are single-seed.
- DQN is still baseline-only in the latest phase (not re-run in strategic continuation stage).
- Run artifacts under `runs/` are large and intentionally ignored from version control.
