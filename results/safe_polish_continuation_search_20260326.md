# Safe-polish continuation search — 2026-03-26

## Goal
Continue improving broad coverage from `runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish/best_model.zip`, with special attention to `chokepoint` and `sneaky_enemies`, while preserving the near-solved easy maps.

## Baseline checked first
Fresh 128-episode deterministic checks on the current leader:

- Broad mixed-eval mean coverage: **0.846203**
- Per-map 128-episode mean coverage:
  - `just_go`: **1.000000**
  - `safe`: **1.000000**
  - `maze`: **0.995763**
  - `chokepoint`: **0.665249**
  - `sneaky_enemies`: **0.519271**
- Per-map average: **0.836057**

## Important trainer fix found
Same-architecture continuations using `init_model_path` were silently keeping the checkpoint's optimizer schedule, so requested low learning rates in continuation configs were **not actually applied**.

### Fix
`rl_coverage/train.py` now reapplies configured continuation overrides after load, including:
- `learning_rate`
- `clip_range`
- `clip_range_vf`
- core scalar PPO attrs like `gamma`, `gae_lambda`, `ent_coef`, `vf_coef`, `batch_size`, `n_steps`

This makes future continuation configs trustworthy.

## New curricula/configs added
- `ppo_strategic_temporal_safe_polish_all_standard_lowlr.toml`
- `ppo_strategic_temporal_safe_polish_hard_anchor.toml`
- `ppo_strategic_temporal_safe_polish_sneaky_anchor.toml`

New map suites added in `rl_coverage/maps.py`:
- `tournament_hard_anchor`
- `tournament_sneaky_anchor`

## Runs attempted
### 1) all-standard low-LR recovery
- Run: `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_all_standard_lowlr`
- Init: current safe-polish leader
- LR: `2e-5`
- Timesteps: `80k`
- Best observed periodic eval: **0.879** mean coverage at ~190k total steps

#### Final broad numbers
- 64-episode final eval mean coverage: **0.853125**
- 128-episode manual broad eval of saved best: **0.847791**

#### Per-map 128-episode eval of saved best
- `just_go`: **1.000000**
- `safe`: **1.000000**
- `maze`: **0.995763**
- `chokepoint`: **0.650020**
- `sneaky_enemies`: **0.519896**
- Per-map average: **0.833136**

#### Read
This run slightly improved mixed broad score but did **not** improve broad hard-map coverage balance:
- `chokepoint`: **-0.015229** vs baseline
- `sneaky_enemies`: **+0.000625** vs baseline
- easy maps: unchanged

### 2) hard-anchor continuation
- Run: `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_hard_anchor`
- Early evals stayed around **0.811–0.824** mean coverage
- Killed early as low-value

### 3) sneaky-anchor continuation
- Run: `runs/tournament_hunt/20260326-144942-ppo_strategic_temporal_safe_polish_sneaky_anchor`
- Early evals stayed around **0.776–0.798** mean coverage
- Killed early as low-value

## Current verdict
No new champion.

The current best broad-balance checkpoint remains:
- `runs/tournament_hunt/20260326-143040-ppo_strategic_temporal_hard_emphasis_safe_polish/best_model.zip`

The all-standard low-LR continuation is interesting as a mixed-suite score smoother, but it is **not** the better coverage champion because the per-map hard coverage average got worse.

## Recommended next move
Now that continuation LR overrides are fixed, the best next experiment is **not** more broad all-standard polish. Instead:

1. start again from the current safe-polish leader;
2. run a **very short chokepoint-tilted anchor curriculum** (not sneaky-heavy) with real low LR;
3. select checkpoints by **per-map broad eval**, not only mixed-suite eval;
4. stop early unless `chokepoint` improves by enough to offset any sneaky variance.

Reason: current evidence says broad mixed-score improvements are mostly coming from distribution effects, while the real missing coverage is still concentrated in `chokepoint` first and `sneaky_enemies` second.
