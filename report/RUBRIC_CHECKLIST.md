# Coverage Tournament Report — Rubric Checklist

## Assignment constraints pulled from the brief
- **Do not modify `coverage-gridworld/coverage_gridworld/env.py`.**
- If environment-side customization is needed, **`custom.py` is the permitted extension point**.
- Adapt the project through **reward shaping** and **observation-space design**.
- Use a suitable **reinforcement learning algorithm**.
- The report must explain the method and compare approaches with plots.

---

## Rubric compliance tracker

### 1) Implement a reinforcement learning algorithm suitable to the problem `[5]`
**Current repo evidence**
- `rl_coverage/train.py` supports **PPO**, **DQN**, and **A2C** through Stable-Baselines3.
- Built-in experiment configs currently use:
  - `DQN` in `configs/experiments/dqn_rgb_baseline.toml`
  - `PPO` in the three PPO configs

**Status**: ✅ Satisfied in code

**Best report evidence to include**
- Why PPO is the primary algorithm for the final agent
- Why DQN is retained as a baseline
- Mention that the task is long-horizon, sparse-ish without shaping, and sensitive to representation quality

---

### 2) Implement at least two distinct observation spaces `[6]`
**Current repo evidence**
- `rl_coverage/observations.py` defines three variants:
  1. `raw_rgb` — flattened normalized RGB grid (**300 features**)
  2. `layered_grid` — 7 semantic channels + 5 scalars (**705 features**)
  3. `frontier_features` — handcrafted compact state summary (**53 features**)

**Status**: ✅ Exceeds minimum

**Best report evidence to include**
- Motivation and expected tradeoffs for each observation space
- Why the observation variants are implemented as wrappers instead of by editing the base env

---

### 3) Implement at least three unique reward functions `[9]`
**Current repo evidence**
- `rl_coverage/rewards.py` defines three variants:
  1. `sparse_coverage`
  2. `dense_coverage`
  3. `survival_coverage`

**Status**: ✅ Satisfied in code

**Best report evidence to include**
- A short description of each reward function
- The behavioral hypothesis behind each one
- A table of reward terms and parameter values used in the experiments

---

### 4) Experiment with each observation space and reward function; generate plots for each experiment `[5]`
**Current repo evidence**
- Built-in configs only cover **4 runs**:
  - `dqn_rgb_baseline`
  - `ppo_frontier_dense`
  - `ppo_frontier_survival`
  - `ppo_layered_dense`
- This does **not** yet give a clean full comparison across all observation/reward combinations.

**Status**: ⚠️ Partially prepared, not yet evidenced

**Missing evidence**
- Completed run artifacts (`summary.json`, `evaluations.json`, `final_evaluation.json`)
- Learning curves and final comparison plots
- A controlled ablation covering all observation spaces and all reward functions

**Safest rubric-compliant plan**
- Run a **3 × 3 PPO ablation matrix** with:
  - Observation ∈ {`raw_rgb`, `layered_grid`, `frontier_features`}
  - Reward ∈ {`sparse_coverage`, `dense_coverage`, `survival_coverage`}
- Keep algorithm, map suite, and evaluation settings fixed for fairness
- Use the DQN baseline as an additional reference, not as the main ablation axis

---

### 5) Train the best possible agent with the proposed approaches and possibly unique training strategies `[5]`
**Current repo evidence**
- Multi-map training is supported through `map_suite`
- Available suites in `rl_coverage/maps.py`:
  - `all_standard`
  - `easy_to_hard`
  - `hard_only`
  - `core_generalization`
- Periodic evaluation + best-model checkpointing already exist in `rl_coverage/callbacks.py`

**Status**: ⚠️ Infrastructure exists, final evidence missing

**Missing evidence**
- A selected “best agent” backed by completed results
- Final evaluation metrics for the chosen best model
- Evidence for any extra strategy such as curriculum maps, longer training, or fine-tuning

**Best candidate strategy right now**
- Current leading starting point from prior design analysis: **PPO + `frontier_features` + `dense_coverage`** on `core_generalization`
- This is still a **rationale-based hypothesis, not yet an empirical winner**
- Backup candidate: **PPO + `frontier_features` + `survival_coverage`** if deaths remain high
- Optional stronger final strategy:
  1. pretrain on `easy_to_hard`
  2. evaluate on `core_generalization`
  3. optionally fine-tune on `sneaky_enemies`

---

### 6) Comprehensive report `[10]`
The brief requires the report to include:
- algorithm description
- observation-space description + rationale
- reward-function description
- training strategy description
- plots of experiment results
- detailed comparison discussion, including **best** and **worst** approach and why

**Status**: ⚠️ Draftable now, final evidence still missing

**Already available for writing**
- Problem description
- Environment rules
- Implementation details of observation/reward wrappers
- Training/evaluation pipeline
- Planned experiment methodology

**Still needed before final submission**
- Actual numerical results
- Plots from logged evaluations
- Final comparison statements backed by evidence

---

## Clean experiment plan for the report

### Controlled ablation block (main rubric evidence)
Use the same setup except for observation/reward:
- **Algorithm:** PPO
- **Policy:** `MlpPolicy`
- **Map suite:** `core_generalization`
- **Evaluation episodes:** 24
- **Deterministic eval:** true
- **Seeds:** ideally 3 seeds per setting if time allows; otherwise 1 seed with limitation stated

| ID | Observation | Reward | Purpose |
|---|---|---|---|
| A1 | raw_rgb | sparse_coverage | weakest/original-style baseline |
| A2 | raw_rgb | dense_coverage | isolate reward shaping benefit on raw input |
| A3 | raw_rgb | survival_coverage | test whether reward alone can rescue raw input |
| B1 | layered_grid | sparse_coverage | isolate representation benefit |
| B2 | layered_grid | dense_coverage | structured-grid balanced setting |
| B3 | layered_grid | survival_coverage | safer structured-grid variant |
| C1 | frontier_features | sparse_coverage | isolate compact feature benefit |
| C2 | frontier_features | dense_coverage | strongest expected all-around setup |
| C3 | frontier_features | survival_coverage | safety-focused compact-feature setup |

### Additional baseline block
- `DQN + raw_rgb + sparse_coverage` on `hard_only` or `sneaky_enemies`
- Use this as a legacy/original-style reference, but do **not** mix it into the fairness claim for the 3 × 3 PPO ablation

### Best-agent block
After the ablation:
- Choose the best PPO configuration by **mean coverage**, then **success rate**, then **death rate**
- Train longer and/or with a stronger map strategy
- Save final evidence from `best_model_metrics.json` and `final_evaluation.json`

---

## Recommended comparison criteria in the writeup
Pull directly from the saved summaries:
- **Mean coverage** (primary metric)
- **Success rate**
- **Death rate**
- **Timeout rate**
- **Mean reward**
- **Mean episode length**

Recommended ordering for “best approach” claim:
1. highest mean coverage
2. highest success rate
3. lower death rate
4. lower variance / more stable learning curve

---

## Current bottom line
The codebase already satisfies the **implementation** parts of the rubric.
The remaining gap is almost entirely **experimental evidence + plots + final comparison writing**.
