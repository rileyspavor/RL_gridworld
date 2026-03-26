# Results Integration Guide

Use this file when concrete experiment outputs arrive from the project/experiment agent.

---

## 1) What artifacts to collect per run
For each completed run, try to gather:
- `config.json`
- `summary.json`
- `final_evaluation.json`
- `evaluations.json`
- `best_model_metrics.json` (if present)
- run directory name

If only one file is available, `summary.json` is the minimum useful input for the final comparison table.

---

## 2) Where each artifact goes in the report

### `summary.json`
Use for the main final-results table:
- mean coverage
- success rate
- death rate
- timeout rate
- mean reward
- mean episode length
- total timesteps
- algorithm / observation / reward labels

### `evaluations.json`
Use for learning-curve plots:
- coverage vs timesteps
- success rate vs timesteps
- death rate vs timesteps
- optionally reward vs timesteps

### `best_model_metrics.json`
Use when discussing the best checkpoint during training.
This is helpful if the final saved model is slightly worse than the best intermediate checkpoint.

### `final_evaluation.json`
Use for the final reported performance of the final saved model.
If there is a discrepancy between final-model performance and best-checkpoint performance, say so explicitly.

---

## 3) Canonical row format for the master table
Fill `RESULTS_TABLE_TEMPLATE.csv` using one row per experiment:
- `experiment_id`
- `algorithm`
- `observation`
- `reward`
- `map_suite`
- `seed`
- `total_timesteps`
- `mean_coverage`
- `success_rate`
- `death_rate`
- `timeout_rate`
- `mean_reward`
- `mean_length`
- `run_dir`
- `notes`

Recommended note examples:
- `best checkpoint better than final model`
- `run stopped early`
- `1 seed only`
- `trained on hard_only instead of core_generalization`

---

## 4) Claim discipline
Until the full comparison exists:
- call `PPO + frontier_features + dense_coverage` the **leading hypothesis** or **best starting configuration**
- do **not** call it the final best method unless the numbers support it

When final results arrive:
- the **best** approach should be selected primarily by mean coverage
- break ties with success rate, then death rate
- mention stability if learning curves clearly differ

---

## 5) Drop-in paragraph templates

### Best method
The best-performing configuration was **[CONFIG]**, which achieved **[MEAN_COVERAGE]** mean coverage, **[SUCCESS_RATE]** success rate, and **[DEATH_RATE]** death rate over **[EPISODES]** evaluation episodes. Relative to the other methods, this suggests that **[OBSERVATION]** provided the most useful decision information and that **[REWARD]** gave the best balance between exploration pressure and safety.

### Worst method
The weakest configuration was **[CONFIG]**, which achieved only **[MEAN_COVERAGE]** mean coverage and showed **[MAIN FAILURE MODE]**. This indicates that **[EXPLANATION]**, limiting the agent's ability to convert local decisions into safe long-horizon coverage.

### Survival-vs-dense comparison
Comparing `dense_coverage` to `survival_coverage`, the main tradeoff was **[TRADEOFF]**. The stronger survival penalties **[REDUCED / DID NOT REDUCE]** deaths, but they also **[SLOWED / DID NOT SLOW]** progress toward full coverage.

### Observation comparison
Across observation spaces, **[OBSERVATION]** performed best because **[EXPLANATION]**. In contrast, **[WORST_OBSERVATION]** underperformed because **[EXPLANATION]**.

---

## 6) Plot checklist once data exists
- [ ] Coverage vs timesteps for the 3×3 PPO matrix
- [ ] Success rate vs timesteps for the 3×3 PPO matrix
- [ ] Death rate vs timesteps for the 3×3 PPO matrix
- [ ] Final comparison table filled from `summary.json`
- [ ] Best-agent paragraph updated with actual numbers
- [ ] Worst-agent paragraph updated with actual numbers
- [ ] Discussion changed from hypothesis language to evidence-backed language

---

## 7) Minimum acceptable final evidence package
If time gets tight, the absolute minimum package for a defensible report is:
1. one completed run for each observation/reward combination in the PPO matrix
2. one final comparison table
3. at least one learning-curve plot family
4. one longer/best-agent run
5. clear acknowledgment of limitations such as single-seed results
