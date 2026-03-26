# Strategic recovery search — 2026-03-26

## Goal
Push beyond the existing temporal-frontier balanced winner by validating the newer strategic temporal feature family on broad evaluation, then continue only the strongest candidate.

## What was checked first
Broad 64-episode deterministic evaluation and per-map sanity were used instead of trusting training-suite scores alone.

### Existing candidate checks
- `runs/tournament_balance/exp_safe_polish_from_balanced_winner`
  - `all_standard` 64-episode eval: mean coverage **0.822**
  - per-map 64-episode mean coverage average: **0.830116**
- `runs/tournament_balance/exp_winner_transfer_lowlr_safe_polish`
  - `all_standard` 64-episode eval: mean coverage **0.819**
  - per-map 64-episode mean coverage average: **0.832954**
- `runs/tournament_hunt/exp_strategic_all_standard`
  - per-map 64-episode mean coverage average: **0.825176**
- `runs/tournament_hunt/exp_strategic_hard_emphasis`
  - `all_standard` 64-episode eval: mean coverage **0.814**
  - per-map 64-episode mean coverage average: **0.840109**
  - strongest hard-map balance among checked candidates

## New experiments

### 1) Wide-net strategic transfer from the balanced temporal winner
- Config: `configs/experiments/ppo_strategic_temporal_winner_transfer_all_standard_wide.toml`
- Idea: modest capacity sweep (`net_arch = [256, 256]`) while transferring into `strategic_temporal_frontier_features`
- Outcome: **rejected early**
  - transfer only matched a tiny policy prefix because the wider architecture changed tensor shapes
  - early eval collapsed to roughly **0.11–0.13** coverage
  - run was terminated rather than wasting more compute

### 2) Strategic hard-emphasis → all-standard recovery continuation
- Config: `configs/experiments/ppo_strategic_temporal_hard_emphasis_all_standard_recovery.toml`
- Init model: `runs/tournament_hunt/exp_strategic_hard_emphasis/best_model.zip`
- Curriculum: switch from `tournament_hard_emphasis` to `all_standard`
- Learning rate: `4e-5`
- Timesteps: `120000` additional steps with `reset_num_timesteps = false`
- Run dir: `runs/tournament_hunt/20260326-134821-ppo_strategic_temporal_hard_emphasis_all_standard_recovery`

#### Best observed training eval during continuation
- around `190k` timesteps total: mean coverage **0.876** on the 24-episode periodic eval
- around `180k` timesteps total: mean coverage **0.874**

#### Broad evaluation of the saved best checkpoint
- `all_standard` 64 deterministic episodes: mean coverage **0.839**, success **0.312**, death **0.219**, timeout **0.469**
- per-map 64 deterministic episodes:
  - `just_go`: **1.000000**
  - `safe`: **0.985507**
  - `maze`: **0.996028**
  - `chokepoint`: **0.678995**
  - `sneaky_enemies`: **0.519792**
- per-map mean coverage average: **0.836064**

## Current read
- The strategic feature family is real: it beats the older temporal winner on per-map balance when paired with the hard-emphasis curriculum.
- The best raw per-map balanced checkpoint checked so far remains:
  - `runs/tournament_hunt/exp_strategic_hard_emphasis/best_model.zip`
  - per-map mean coverage average: **0.840109**
- The new recovery continuation is a solid alternative when you want a slightly stronger broad `all_standard` score without giving up too much hard-map strength.
- Naive architecture widening is **not** plug-compatible with the current transfer trick; it needs either same-shape continuation or a more explicit transfer path.

## Recommended next move
Start from `runs/tournament_hunt/exp_strategic_hard_emphasis/best_model.zip` and run a **short safe-polish / all-standard polish** continuation with the same architecture, using checkpoint selection by broad eval. Do **not** spend more time on mismatched-architecture transfer unless transfer logic is upgraded beyond the current prefix copy.
