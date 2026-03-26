# Experiment Inbox

Run log for the submission snapshot.

---

## 2026-03-25 — Controlled 3×3 PPO observation/reward sweep
- Scope: `core_generalization`, 9 PPO runs, 50k steps each, seed 21.
- Winner: `frontier_features + dense_coverage`.
- Key metrics: mean coverage **0.458**, success **0.000**, death **0.667**.
- Weakest combo: `raw_rgb + sparse_coverage` at **0.121** mean coverage.
- Artifacts: `results/observation_reward_sweep/*`.

## 2026-03-25 — Long-run frontier baseline
- Run: `runs/20260325-192838-ppo_frontier_dense`.
- Final deterministic eval (24 eps): coverage **0.523**, success **0.250**, death **0.667**, timeout **0.083**.
- Best checkpoint around 325k (16 eps): coverage **0.557**.
- Outcome: strong baseline, hard-map gap remains.

## 2026-03-25 — Hard-only fine-tune branch
- Winner for that phase: `runs/finetune/20260325-195113-ppo_frontier_dense_hard_finetune/best_model.zip`.
- Improved mean coverage over baseline in broad checks, but still weak on hardest maps.
- Documented in: `results/tournament_agent_search_20260325.md`.

## 2026-03-26 — Strategic temporal continuation search
- Strategic family validated and pushed with continuation curricula.
- Broad mixed-suite scalar leader (128 eps):
  - `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_all_standard_lowlr/best_model.zip`
  - mean coverage **0.848**.
- Best hard-map-balance checkpoint:
  - `runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish/best_model.zip`
  - stronger `chokepoint` mean coverage (**0.665** vs **0.650**), per-map average **0.836**.

## 2026-03-26 — Trainer continuation override fix
- Issue found: low-LR continuation configs could silently retain checkpoint optimizer schedule.
- Fix: `rl_coverage/train.py` now reapplies continuation overrides after load.
- Impact: continuation experiments now honor configured `learning_rate`, clip settings, and key PPO scalars.
