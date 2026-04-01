# Coverage Gridworld RL Project Report Draft

## Title
Coverage Gridworld with PPO: Observation-Space Design, Reward Shaping, and Experimental Comparison

## Authors


## 1. Introduction and Problem Formulation

The goal of this project is to train a reinforcement learning agent to cover all reachable cells in a 10x10 grid environment while avoiding detection by rotating enemies. The environment contains walls, unexplored cells, and enemies with a field of view extending up to four cells in the direction they are facing. At every step, the enemies rotate, which makes the task both spatial and temporal: the agent must not only decide where to go, but also when it is safe to move.

The assignment required adapting the environment through reward shaping and observation-space design without modifying the core environment implementation. In this project, `env.py` was intentionally left untouched, and all customization was handled through `custom.py`, feature-engineered observation spaces, and configurable reward functions.

### 1.1 Action Space
The action space is **discrete** with five actions:
- `0`: move left
- `1`: move down
- `2`: move right
- `3`: move up
- `4`: stay in place

This is a small discrete decision space, so a policy-gradient method for discrete control is appropriate.

### 1.2 State and Observation Space
The full environment state includes:
- the current map layout
- the agent position
- the set of covered and uncovered cells
- enemy positions
- enemy orientations
- future consequences of enemy rotation

The agent does not directly consume a symbolic state representation. Instead, custom observation spaces were designed to summarize the information most useful for learning.

Two observation spaces were used in the clean assignment experiment matrix:

1. **`frontier_features` (40 dimensions)**  
   A compact feature vector that summarizes agent position, coverage progress, nearby occupancy, directional movement affordances, and frontier-distance information.

2. **`temporal_frontier_features` (98 dimensions)**  
   Extends the frontier representation with short-horizon enemy-visibility forecasts and compact enemy state summaries.

Both observation spaces are implemented as continuous Gymnasium `Box` spaces.

### 1.3 Reward Scheme
Three distinct reward functions were designed and compared:
- `sparse_coverage`
- `dense_coverage`
- `survival_coverage`

Each reward function uses the same environment information but shapes learning differently. The main objective was to determine which combination of reward formulation and observation design best improves coverage while reducing unsafe behavior.

---

## 2. Methodology and Algorithm Description

### 2.1 Algorithm Choice
This project used **Proximal Policy Optimization (PPO)** from Stable-Baselines3 with an `MlpPolicy` network. PPO was chosen because:
- it is stable and widely used,
- it works well in discrete-action environments,
- it supports vectorized training,
- it is simple to configure and reproduce.

Because the observation spaces were designed as compact feature vectors rather than images, a multilayer perceptron policy was sufficient.

### 2.2 PPO Configuration
All six assignment-matrix experiments used the same PPO backbone so the comparison would isolate the effects of observation-space and reward-function design.

Shared PPO hyperparameters:
- Policy: `MlpPolicy`
- Learning rate: `3e-4`
- `n_steps = 1024`
- `batch_size = 256`
- `gamma = 0.995`
- `gae_lambda = 0.95`
- `clip_range = 0.2`
- `ent_coef = 0.01`
- `vf_coef = 0.5`
- Number of parallel environments: `4`
- Total training timesteps per run: `180,000`

Evaluation settings:
- periodic evaluation every `20,000` timesteps
- `10` deterministic episodes during training evaluation
- `20` deterministic episodes for final comparison

### 2.3 Implementation Architecture
The code was organized so the course environment remained intact while the RL logic stayed modular.

- `coverage-gridworld/coverage_gridworld/env.py`  
  Original environment, intentionally unchanged.

- `coverage-gridworld/coverage_gridworld/custom.py`  
  Assignment-approved customization hook. This file delegates observation and reward logic to configurable helper modules.

- `project_rl/observations.py`  
  Defines the custom observation spaces.

- `project_rl/rewards.py`  
  Defines the reward functions.

- `project_rl/training.py`, `metrics.py`, `callbacks.py`, `plotting.py`  
  Handle training, evaluation, logging, summaries, and plots.

This structure made it possible to run multiple experiments cleanly without hardcoding all logic in a single file.

### 2.4 Experiment Design
The main experimental design was a clean **1 x 2 x 3** matrix:
- **1 algorithm:** PPO
- **2 observation spaces:** `frontier_features`, `temporal_frontier_features`
- **3 reward functions:** `sparse_coverage`, `dense_coverage`, `survival_coverage`

This produced six directly comparable training runs.

All assignment-matrix runs used the `all_standard` map suite so each configuration was trained and evaluated on the same standard family of maps.

---

## 3. Observation Spaces and Design Rationale

### 3.1 Frontier Features
The `frontier_features` observation was designed to be compact, stable, and easy for PPO to learn from. Instead of using the raw RGB grid directly, it summarizes the aspects of the state most relevant to exploration:
- normalized agent position
- current coverage ratio
- local neighborhood semantics
- blocked or available movement directions
- directional frontier-distance information
- nearest frontier location cues

#### Rationale
This representation focuses on the exploration problem directly. The agent needs to know where unexplored cells remain and which directions are currently navigable. By compressing the grid into a smaller task-relevant vector, PPO can learn more efficiently than with a raw flattened grid.

### 3.2 Temporal Frontier Features
The `temporal_frontier_features` observation extends the compact frontier representation with near-future safety information. In addition to the frontier-focused signals, it includes:
- future enemy observation masks for candidate actions
- safe-action counts under future enemy rotations
- compact enemy orientation summaries
- relative enemy-position information

#### Rationale
This environment is not purely about exploration; it is also about timing. Because enemies rotate each step, a policy that only understands the current layout may still move into danger one step later. The temporal observation was designed to help the agent anticipate those short-term hazards.

### 3.3 Compatibility Observation for `comp_model_80`
Outside the clean assignment matrix, a third observation path was reconstructed:
- **`strategic_temporal_frontier_features` (142 dimensions)**

This observation was not part of the official 2x3 matrix. It exists only to replay and evaluate the recovered checkpoint `comp_model_80`, which was trained with a richer handcrafted observation vector.

---

## 4. Reward Functions and Design Rationale

Three reward functions were implemented and compared.

### 4.1 Sparse Coverage
This reward is the simplest baseline. It includes:
- a step penalty
- a reward for covering a new cell
- a success bonus
- a death penalty
- a timeout penalty

#### Rationale
The purpose of this reward is to provide a minimal learning signal centered on the task objective. It serves as a baseline against which denser reward shaping can be compared.

### 4.2 Dense Coverage
This reward adds more continuous shaping, including:
- step penalty
- reward for covering new cells
- no-progress penalty
- coverage-ratio shaping
- success bonus
- speed bonus
- death and timeout penalties

#### Rationale
Sparse rewards can make learning slower because useful feedback is rare. Dense reward shaping was introduced to provide the policy with more frequent information about whether it is making meaningful progress.

### 4.3 Survival Coverage
This reward keeps the coverage objective but adds stronger safety-aware shaping:
- step penalty
- reward for new coverage
- coverage-ratio shaping
- no-progress penalty
- penalty for observed / risky positions
- stronger death penalty
- success and speed bonuses
- timeout penalty

#### Rationale
In this environment, good behavior is not only “cover more cells,” but “cover more cells without getting caught.” This reward was designed to align the learning signal more closely with that combined objective.

---

## 5. Results

The final 2x3 PPO experiment matrix was evaluated on **20 deterministic episodes per configuration**.

| Observation | Reward | Mean Coverage | Success Rate | Death Rate | Mean Reward | Mean Length |
|---|---|---:|---:|---:|---:|---:|
| temporal_frontier_features | survival_coverage | **0.610** | **0.30** | 0.45 | 54.924 | 157.05 |
| temporal_frontier_features | sparse_coverage | 0.581 | 0.30 | **0.40** | 42.412 | 188.80 |
| temporal_frontier_features | dense_coverage | 0.572 | 0.30 | 0.45 | **64.459** | 158.10 |
| frontier_features | survival_coverage | 0.551 | 0.20 | 0.55 | 47.874 | **154.20** |
| frontier_features | dense_coverage | 0.518 | 0.20 | 0.50 | 60.017 | 177.60 |
| frontier_features | sparse_coverage | 0.504 | 0.20 | 0.55 | 35.679 | 157.15 |

Generated report artifacts are available in:
- `results/plots/leaderboard.csv`
- `results/plots/mean_coverage_by_combo.png`
- `results/plots/success_rate_by_combo.png`
- `results/plots/mean_reward_by_combo.png`
- `results/plots/learning_curves.png`
- `results/plots/coverage_heatmap.png`

---

## 6. Discussion

### 6.1 Best Overall Assignment Configuration
The strongest clean assignment result was:
- **Observation:** `temporal_frontier_features`
- **Reward:** `survival_coverage`

This combination achieved the highest mean coverage (**0.610**) and tied for the highest success rate (**0.30**).

### 6.2 Why This Combination Worked Best
This outcome makes sense given the structure of the task.

The agent must:
1. find unexplored frontier cells efficiently, and
2. avoid enemy observation over time.

The temporal observation improves the second part by giving the policy short-horizon safety information, while the survival-oriented reward penalizes behavior that is reckless or too exposure-prone. Together, they produce a more balanced exploration policy.

### 6.3 Worst-Performing Configuration
The weakest clean assignment result was:
- **Observation:** `frontier_features`
- **Reward:** `sparse_coverage`

This configuration had the lowest mean coverage (**0.504**) and one of the highest death rates (**0.55**).

### 6.4 Why It Performed Worse
This combination provided the least guidance:
- the observation did not include explicit temporal danger information,
- the reward was only lightly shaped.

As a result, PPO could still learn some useful movement behavior, but it had less information about when moves would become unsafe and received weaker signals for improving survival.

### 6.5 Observation-Space Comparison
Across all three reward functions, `temporal_frontier_features` outperformed `frontier_features` on mean coverage and matched or exceeded it on success rate. This suggests that temporal awareness is genuinely useful in this environment and not just an implementation detail.

### 6.6 Reward-Function Comparison
Across both observation spaces, `survival_coverage` produced the strongest overall task performance. `dense_coverage` sometimes produced higher raw mean reward, but not always the highest mean coverage. This is important because it shows that the numerically largest reward is not necessarily the reward formulation best aligned with the assignment objective.

---

## 7. Best-Agent Extension and Recovered Checkpoint

After the clean assignment matrix was completed, an existing checkpoint named `comp_model_80` was reverse-engineered and made runnable by reconstructing its expected 142-dimensional observation pipeline.

This checkpoint was **not** part of the official 2x3 assignment comparison. It was kept as a separate compatibility benchmark.

Recovered checkpoint evaluation (`runs/matrix/comp_model_80_play`):
- Mean coverage: **0.848**
- Success rate: **0.45**
- Death rate: **0.20**
- Mean reward: **79.311**
- Mean episode length: **225.65**

The recovered checkpoint clearly outperformed the clean assignment-matrix winner. This suggests that it likely came from a more engineered training pipeline with richer handcrafted features and additional tuning beyond the clean matrix experiments.

---

## 8. Conclusion

This project showed that both observation-space design and reward shaping substantially affect PPO performance in Coverage Gridworld.

The main conclusions are:
- compact handcrafted features are sufficient for PPO to learn useful policies,
- temporal danger information improves performance compared with purely spatial frontier features,
- safety-aware reward shaping improves the exploration/survival trade-off,
- the best clean assignment configuration was `temporal_frontier_features + survival_coverage`.

Overall, the project achieved the assignment objective of adapting the environment through reward shaping and observation-space design while keeping the environment core unchanged.

---

## 9. Submission Notes

For the final hand-in, include:
- the Python project files
- the trained model ZIP submitted separately
- the generated plots referenced in the Results section
- final author names
- figure captions and formatting required by the course rubric

In the cleaned submission repository:
- full artifacts were retained for the best clean assignment run,
- full artifacts were retained for the recovered `comp_model_80` compatibility run,
- the remaining matrix runs were reduced to compact summary artifacts to keep the repository easier to navigate while preserving the final comparison results.
