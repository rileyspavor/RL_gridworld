# Results Directory Guide

This directory mixes generated artifacts and written summaries.

## Authoritative for submission claims
- `rl_project_findings.md` — consolidated submission snapshot.
- `observation_reward_sweep/leaderboard.csv` — canonical 3×3 PPO matrix table.
- `observation_reward_sweep/summary.md` — quick narrative of matrix outcome.
- `strategic_recovery_search_20260326.md` — strategic feature recovery phase notes.
- `safe_polish_continuation_search_20260326.md` — continuation phase notes.
- `tournament_agent_search_20260325.md` — earlier tournament-agent search notes.

## Generated plots
- `observation_reward_sweep/*.png` — matrix comparison figures used in report.
- `plots_smoke/*.png` — smoke/demo plotting output (not used for final claims).

## Dashboard export
- `dashboard_snapshot.json` — point-in-time API snapshot for dashboard integration.

## Rebuild hints

```bash
python -m rl_coverage.plot_runs --runs-dir runs/sweep --output-dir results/observation_reward_sweep
python -m rl_coverage.dashboard_api snapshot --output results/dashboard_snapshot.json
```

## Notes
- `runs/` holds the detailed raw run artifacts referenced by these summaries.
- `runs/` is ignored in git by design due size; archive separately for external submission packages when needed.
