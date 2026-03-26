# Coverage Tournament Report Draft

## Title
**Coverage Tournament: Reinforcement Learning for Safe Exploration in Coverage Gridworld**

## Authors
Riley Spavor et al.  
Queen's University

---

## Abstract
This project studies reinforcement learning for a grid-based coverage task in which an agent must explore all reachable cells while avoiding rotating enemies with line-of-sight fields of view. The main goal was not to redesign the environment itself, but to improve agent performance through two practical RL levers: **observation-space design** and **reward shaping**. To do this, the project keeps the base environment unchanged and instead applies modular wrappers that swap in multiple observation representations and reward functions. The implementation supports several RL algorithms, with PPO used as the main algorithm for controlled experiments and DQN retained as a baseline closer to the original starter direction. We compare raw RGB observations, a semantically layered grid representation, and a compact handcrafted frontier-based feature representation under sparse, dense, and survival-oriented reward functions. The report evaluates how representation quality and reward design affect learning speed, coverage, success rate, and failure modes, and then identifies the strongest final agent configuration.

> **Fill after experiments:** replace the last two sentences with a short summary of the actual best result.

---

## 1. Problem Setup
The task is a 10×10 gridworld coverage problem. The agent starts in the top-left corner and must cover all reachable cells. Some cells contain walls, and some contain enemies. Each enemy rotates counter-clockwise every step and observes cells in a forward field of view of length four, blocked by walls and other enemies. If the agent is seen, the episode ends. An episode also ends if all coverable cells are explored or the 500-step budget expires.

This task is challenging for two reasons. First, the objective is **long-horizon**: the agent must make decisions that improve eventual total coverage rather than optimize only short-term safety. Second, the environment contains **partial tactical danger structure**: the same move can be good or catastrophic depending on enemy orientation and line of sight. This makes the task sensitive to how state information is encoded and how rewards assign credit to progress, hesitation, looping, and death.

A key project constraint was that `coverage_gridworld/env.py` could not be modified. If any environment-side customization was needed, `custom.py` was the intended extension point. In practice, the project's main changes were implemented outside the base environment through wrappers and experiment code so the underlying task dynamics stayed fixed.

---

## 2. Algorithm Choice
The project infrastructure supports PPO, DQN, and A2C through Stable-Baselines3. For the main experiments, **PPO** was chosen as the primary algorithm because it is a strong default for on-policy control in discrete environments with shaped rewards and nontrivial exploration structure. PPO is especially suitable here for three reasons:

1. **Long-horizon behavior:** the task rewards consistent exploration over many steps.
2. **Shaped rewards:** PPO tends to work well when dense but imperfectly shaped reward signals are used.
3. **Feature-friendly training:** the compact and layered observation variants are naturally compatible with an MLP policy.

**DQN** is retained as a baseline because it is closer to the original starter setup and gives a useful reference for how much the redesigned pipeline improves over a simpler value-based approach on raw flattened observations.

### Report insert point
- **Main algorithm:** PPO
- **Baseline algorithm:** DQN
- **Optional note:** A2C exists in the codebase but was not the main focus of the current comparison

---

## 3. Observation Spaces and Rationale
Observation design is one of the core variables in this project. The environment produces a color-coded grid, but that raw form is not necessarily the easiest representation for a policy network to learn from. To compare alternatives fairly, observation transformations are implemented as wrappers in `rl_coverage/observations.py`.

### 3.1 Raw RGB (`raw_rgb`)
This representation flattens the 10×10×3 RGB grid into a 300-dimensional vector normalized to `[0, 1]`.

**Why include it:**
- It is the closest representation to the original environment output.
- It provides a low-assumption baseline.
- It shows what happens when the model must infer the meaning of colors and spatial structure mostly on its own.

**Expected strengths:**
- Minimal manual feature engineering.
- Preserves all rendered information.

**Expected weaknesses:**
- Weak inductive bias for MLPs.
- Harder credit assignment because semantic categories are encoded indirectly through color.
- Flattening removes explicit locality and structure.

### 3.2 Layered Grid (`layered_grid`)
This representation converts the board into seven binary semantic channels: unexplored, observed-unexplored, explored, observed-explored, walls, enemies, and agent, then appends five scalar features. The total feature count is 705.

**Why include it:**
- It keeps global board information while making semantics explicit.
- It removes the need for the network to learn the meaning of colors from scratch.
- It remains general-purpose and does not hard-code a single exploration strategy.

**Expected strengths:**
- Better state interpretability than raw RGB.
- Rich spatial coverage information.
- More faithful to the full map than a compact handcrafted vector.

**Expected weaknesses:**
- Higher dimensional than the other options.
- Still flattened for an MLP, so spatial adjacency is not exploited as efficiently as a CNN-based 2D policy would.

### 3.3 Frontier Features (`frontier_features`)
This is a 53-dimensional handcrafted feature vector designed specifically for coverage behavior. It includes:
- normalized agent position
- coverage ratio and cells remaining
- steps remaining
- immediate move affordances for each action
- local neighborhood categories
- directional free-run and frontier-distance signals
- nearest frontier offsets
- quadrant frontier density
- reachable frontier distance

**Why include it:**
- The task is not generic image understanding; it is structured exploration.
- A compact feature vector can expose the most decision-relevant information directly.
- It is expected to reduce sample complexity and improve learning stability for MLP policies.

**Expected strengths:**
- Strong inductive bias toward exploration and safe movement.
- Lower dimensionality than both grid-based alternatives.
- Likely fastest to learn when paired with effective reward shaping.

**Expected weaknesses:**
- More hand-designed than the other approaches.
- May discard useful global layout detail that a richer spatial representation could exploit later.

### 3.4 Why wrappers were used
Observation variants were implemented outside the environment so the same underlying dynamics could be reused unchanged. This matters for the assignment because it respects the rule against modifying `env.py` while also improving experimental fairness: only the observation interface changes, not the transition logic.

---

## 4. Reward Functions and Rationale
Reward shaping is the second core project variable. The reward variants are implemented in `rl_coverage/rewards.py` as environment wrappers that use the base environment's info dictionary.

### 4.1 Sparse Coverage (`sparse_coverage`)
This is the simplest baseline reward:
- small per-step penalty
- reward for covering a new cell
- completion bonus
- death penalty
- timeout penalty

**Rationale:**
This version provides a reference point close to the original project spirit. It tests whether a policy can learn mostly from task-level progress rather than detailed shaping.

**Hypothesis:**
It should be the hardest reward to optimize, especially with weaker observations, because it gives limited guidance about revisits, loops, and risky local behavior.

### 4.2 Dense Coverage (`dense_coverage`)
This reward adds several shaping terms on top of coverage progress:
- per-step penalty
- new-cell reward
- coverage-progress weight
- invalid-move penalty
- stay penalty
- revisit penalty
- loop penalty
- observed-cell penalty
- success bonus
- death and timeout penalties

**Rationale:**
The task requires more than just eventually finding new cells. The agent also needs pressure against wasting time, repeating positions, standing still, and walking into dangerous observed cells. This reward is intended to create a balanced signal that promotes efficient and safe exploration.

**Hypothesis:**
This should be the strongest default reward because it improves credit assignment without making survival the only priority.

### 4.3 Survival Coverage (`survival_coverage`)
This variant starts from the dense reward and makes the safety and anti-stall terms more aggressive. It also includes an optional coverage-ratio bonus.

**Rationale:**
Some policies may learn to chase coverage greedily and die often. This variant tests whether stronger punishment for dangerous or stagnant behavior improves performance in maps with multiple enemy chokepoints.

**Hypothesis:**
It should reduce catastrophic deaths, but it may also become over-conservative and therefore slightly slower to finish full coverage.

### 4.4 Why reward wrappers were used
As with observation spaces, reward shaping was applied through wrappers rather than by editing the environment. This preserves the base task definition and makes the comparison cleaner: the environment dynamics stay fixed while the learning signal changes.

---

## 5. Training Strategy and Methodology
The project now separates environment logic from experimentation. Training is run through `rl_coverage/train.py`, which loads a TOML config, builds a wrapped environment, trains an SB3 model, periodically evaluates it, and saves artifacts.

### 5.1 Environment and map strategy
The codebase includes five standard maps:
- `just_go`
- `safe`
- `maze`
- `chokepoint`
- `sneaky_enemies`

For multi-map training, the experiment layer defines named map suites. The most important suite for the main report is `core_generalization`, which cycles over:
- `safe`
- `maze`
- `chokepoint`
- `sneaky_enemies`

Using a map suite instead of a single fixed map helps reduce overfitting and makes the evaluation more about general exploration behavior than about memorizing one layout.

### 5.2 Evaluation protocol
Training uses a periodic evaluation callback that writes:
- `latest_evaluation.json`
- `evaluations.json`
- `best_model.zip`
- `best_model_metrics.json`

At the end of training, each run also saves:
- `final_model.zip`
- `final_evaluation.json`
- `summary.json`
- `config.json`

The main comparison metrics are:
- mean coverage
- success rate
- death rate
- timeout rate
- mean reward
- mean episode length

### 5.3 Fair-comparison design
For the main observation/reward comparison, the fairest setup is to hold the following fixed:
- algorithm: PPO
- policy: `MlpPolicy`
- map suite: `core_generalization`
- evaluation protocol: same episode count and deterministic setting
- training length: same number of timesteps

This isolates the effect of the observation and reward choices.

### 5.4 Best-agent training strategy
After the controlled comparison identifies the most promising configuration, the final best-agent experiment can use a stronger training strategy such as:
- longer PPO training
- curriculum-style suite (`easy_to_hard`)
- final fine-tuning on `sneaky_enemies`

This final step satisfies the project requirement to train the strongest agent possible using the proposed approaches.

---

## 6. Experiment Matrix

### 6.1 Main ablation matrix (recommended)
The report should present the main results as a controlled 3 × 3 PPO matrix.

| Experiment | Observation | Reward | Expected role |
|---|---|---|---|
| A1 | raw_rgb | sparse_coverage | weakest baseline |
| A2 | raw_rgb | dense_coverage | effect of reward shaping alone |
| A3 | raw_rgb | survival_coverage | safety shaping on weak representation |
| B1 | layered_grid | sparse_coverage | effect of better representation alone |
| B2 | layered_grid | dense_coverage | structured-grid balanced setup |
| B3 | layered_grid | survival_coverage | structured-grid safer setup |
| C1 | frontier_features | sparse_coverage | compact features without strong shaping |
| C2 | frontier_features | dense_coverage | expected strongest overall |
| C3 | frontier_features | survival_coverage | compact features with stronger safety pressure |

### 6.2 Additional algorithm baseline
A separate baseline experiment should be included:
- `DQN + raw_rgb + sparse_coverage`

This is useful historically, but it should be discussed separately from the clean PPO ablation because otherwise algorithm choice becomes confounded with observation and reward choice.

---

## 7. Planned Results Section Structure

### 7.1 Learning curves
**Figure 1.** Mean coverage vs. timesteps for all main PPO experiments.  
**Figure 2.** Success rate vs. timesteps for all main PPO experiments.  
**Figure 3.** Death rate vs. timesteps for all main PPO experiments.

**What to discuss:**
- which setups learn fastest
- which setups plateau early
- which setups are unstable
- whether stronger safety shaping lowers death at the cost of slower coverage

### 7.2 Final evaluation table
**Table 1.** Final evaluation summary for all experiments.

Suggested columns:
- experiment id
- algorithm
- observation
- reward
- mean coverage
- success rate
- death rate
- timeout rate
- mean reward
- mean episode length

### 7.3 Best and worst approaches
This subsection should answer the rubric directly:
- Which approach was the **best**, and why?
- Which approach was the **worst**, and why?

### 7.4 Qualitative behavior
If possible, include a short qualitative figure or screenshot sequence from manual play or rollout videos showing how the best agent behaves differently from the weakest baseline.

---

## 8. Results Discussion Framework

### 8.1 Comparison criteria
The strongest method should be judged primarily by:
1. **mean coverage**
2. **success rate**
3. **death rate**
4. learning stability / consistency

### 8.2 Anticipated interpretation
A likely interpretation is:
- `raw_rgb` underperforms because the policy must learn color semantics and exploration structure indirectly.
- `layered_grid` improves learning because it exposes semantic categories explicitly while preserving full-map information.
- `frontier_features` performs best when the handcrafted representation aligns well with the decision structure of the task.
- `dense_coverage` is the most balanced reward because it encourages progress without making the agent excessively timid.
- `survival_coverage` may reduce death rate but can become conservative.
- `sparse_coverage` is hardest because it offers the weakest intermediate guidance.

> Replace the above with evidence-backed claims once the experiments finish.

### 8.3 Best-approach framing template
**Template paragraph:**  
The best-performing approach was **[INSERT CONFIG]**, which achieved **[INSERT MEAN COVERAGE]** mean coverage and **[INSERT SUCCESS RATE]** success rate while maintaining a **[INSERT DEATH RATE]** death rate. This result suggests that **[INSERT EXPLANATION]**. In particular, the combination of **[observation property]** and **[reward-shaping property]** helped the agent balance exploration efficiency with survival.

### 8.4 Worst-approach framing template
**Template paragraph:**  
The weakest approach was **[INSERT CONFIG]**, which achieved only **[INSERT MEAN COVERAGE]** mean coverage and showed **[INSERT FAILURE MODE]**. This likely happened because **[INSERT EXPLANATION]**, making it difficult for the policy to assign credit or recognize safe productive movement.

---

## 9. Current Evidence Available Before Full Runs
The codebase already establishes several strong implementation contributions:
- three observation variants
- three reward variants
- modular wrappers instead of environment edits
- multi-map training support
- periodic evaluation and saved experiment artifacts
- comparison tooling through `rl_coverage.compare`

What is still missing for the final report is not the method description, but the **quantitative evidence**:
- completed runs
- plots
- final metrics table
- best/worst comparison backed by data

At the time of this draft, the strongest reported setup is still only a **design-driven starting hypothesis**: PPO with `frontier_features` and `dense_coverage` on `core_generalization`. That claim should be presented carefully until the experiment matrix confirms or overturns it.

---

## 10. Limitations to Acknowledge
Depending on what experiments are completed, the discussion may need to note:
- limited seed count
- incomplete full-factorial comparison if time prevents all 9 PPO combinations
- use of MLPs on flattened grid features rather than 2D CNN architectures
- aggregation across a map suite rather than per-map breakdown

These do not invalidate the project, but they should be stated clearly if they apply.

---

## 11. Conclusion
This project reframes the Coverage Tournament as a practical RL design problem: instead of changing the environment rules, it improves learning through better state representation, better reward shaping, and cleaner experimentation. The modular pipeline now makes it possible to compare multiple observation spaces and reward functions systematically, train stronger agents than the original starter code could support, and produce a report grounded in both implementation design and empirical evidence.

> Final sentence to fill after results: summarize the best approach and why it won.

---

## Appendix A. Concrete Evidence to Insert Later
- `summary.json` from each completed run
- `evaluations.json` learning-curve data
- `best_model_metrics.json` for best checkpoints
- `final_evaluation.json` for final reported numbers
- output of `python -m rl_coverage.compare --runs-dir runs`

## Appendix B. Plot checklist
- [ ] Coverage vs timesteps
- [ ] Success rate vs timesteps
- [ ] Death rate vs timesteps
- [ ] Final metric comparison table
- [ ] Best-agent qualitative screenshot/GIF (optional but helpful)
