# Experiment Plan

## Main comparisons

1. Q-Learning vs DQN
2. RL vs baselines
3. Different question budgets
4. Different user profile mixes

## Core scripts

- Run baselines:
  - `python -m experiments.run_baselines`
- Run Q-Learning:
  - `python -m experiments.run_q_learning`
- Run DQN:
  - `python -m experiments.run_dqn`
- Run full comparison sweep:
  - `python -m experiments.run_comparison_suite`
- Compare all saved outputs:
  - `python -m experiments.compare_results`

## Suggested matrix

- Question budgets: `0, 1, 2, 3`
- Profiles: each single profile plus mixed defaults (`--profile-mix`)
- Keep seeds fixed per sweep for reproducibility

## Metrics tracked

- average cumulative reward
- acceptance rate
- skip rate
- abandonment rate
- average session length
- average questions asked
- question efficiency

## Output format

Each runner saves json in `results/`.
Comparison script aggregates json files into:
- `comparison_summary.csv`
- `comparison_summary.json`
