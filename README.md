# Budgeted Interactive Movie Recommender Using Reinforcement Learning

This repository implements a custom RL environment where an agent decides whether to ask a preference question or recommend a movie genre under a limited question budget.

## Project purpose

The core research question is: can an RL policy learn when to ask vs recommend to improve long-term recommendation outcomes while avoiding user friction?

This project focuses on:
- custom environment design
- simulated user behavior
- reward shaping for interaction efficiency
- applying existing RL algorithms (Q-Learning, DQN)
- comparing learned policies to simple baselines

## Environment overview

Each episode is a short recommendation session.

At each step, the agent chooses one action:
- ask a clarification question, or
- recommend a genre

The user simulator has hidden preferences and profile-specific behavior. Episodes terminate at max steps or earlier if the user leaves.

## Action space

Question actions:
- `0`: ask familiar vs exploratory
- `1`: ask serious vs light
- `2`: ask fast-paced vs calm

Recommendation actions:
- `3`: recommend Action
- `4`: recommend Comedy
- `5`: recommend Drama
- `6`: recommend Sci-Fi
- `7`: recommend Documentary

## State representations

Two observation modes are supported:
- `tabular`: discrete state id for Q-Learning
- `vector`: dense feature vector (shape 17) for DQN

State features include session step, budget, engagement, belief over genres, uncertainty, last action type, recent outcome, and repetition level.

## Reward intuition

The reward combines:
- positive reward for accepted recommendations
- small continuation bonus for good interactions
- penalties for skip, abandonment, and excessive repetition
- small cost for asking questions
- extra penalty for asking after budget exhaustion

Reward values are configurable in `env/reward_logic.py`.

## Algorithms used

Required:
- tabular Q-Learning
- DQN using Stable Baselines3

Optional:
- PPO (included as an optional script only after core features)

## Installation

From repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start (smoke run)

Use small settings to quickly verify the project is working end to end:

```bash
python -m experiments.run_baselines --episodes 20 --seed 7
python -m experiments.run_q_learning --train-episodes 200 --eval-episodes 50 --seed 7
python -m experiments.run_dqn --total-timesteps 2000 --eval-episodes 50 --seed 7
python -m experiments.compare_results --results-root results
```

## How to run baselines

Run all baselines:

```bash
python -m experiments.run_baselines --episodes 300 --question-budget 2 --seed 7
```

Use mixed-profile sampling:

```bash
python -m experiments.run_baselines --episodes 300 --profile-mix action_focused,balanced_viewer,novelty_seeking,question_sensitive
```

Run a single baseline:

```bash
python -m baselines.always_recommend --episodes 200
```

## How to train Q-Learning

```bash
python -m experiments.run_q_learning --train-episodes 3000 --eval-episodes 300 --seed 7
```

Outputs:
- model: `results/models/q_learning.pkl`
- metrics: `results/q_learning_metrics.json`

## How to train DQN

```bash
python -m experiments.run_dqn --total-timesteps 40000 --eval-episodes 300 --seed 7
```

Outputs:
- model: `results/models/dqn_model.zip` (SB3 suffix)
- metrics: `results/dqn_metrics.json`

## How to run comparisons

Aggregate all saved result json files:

```bash
python -m experiments.compare_results --results-root results/sweeps
```

Run the full sweep (budgets + profile settings + baselines + Q-Learning + DQN):

```bash
python -m experiments.run_comparison_suite --budgets 0,1,2,3 --profile-settings mixed,action_focused,balanced_viewer,novelty_seeking,question_sensitive --seed 7
```

Outputs:
- `results/sweeps/comparison_summary.csv`
- `results/sweeps/comparison_summary.json`

## Tests

Run basic tests:

```bash
pytest
```

## Dashboard Demo

Launch the executive-style multipage product dashboard:

```bash
streamlit run dashboard/app.py
```

Pages included:
- Executive Overview
- Product Demo
- Results and Comparisons
- Technical Credibility

What it uses by default:
- `results/manual_sweep/comparison_summary.csv` (preferred)
- `results/manual_check/comparison_summary.csv` (fallback)
- JSON comparison files if CSV files are unavailable

Optional live-demo model loading:
- newest `results/**/q_learning*.pkl` for Q-Learning policy replay
- newest `results/**/dqn_model*.zip` for DQN policy replay

If comparison/model files are missing, the dashboard shows warnings and keeps running with available data.

## Repository structure

- `env/`: user simulator, reward logic, state encoder, Gymnasium environment
- `baselines/`: non-learning policy baselines
- `agents/`: Q-Learning and DQN/PPO training utilities
- `experiments/`: runners and results comparison tooling
- `dashboard/`: Streamlit executive/demo dashboard
- `tests/`: basic behavior tests
- `docs/`: MDP, user profiles, and experiment plan docs

## Submission status

Complete (core requirements):
- custom Gymnasium environment and user simulator
- baseline policies
- tabular Q-Learning
- DQN with Stable Baselines3
- experiment runners and comparison outputs
- tests and implementation-aligned documentation

Optional:
- PPO training/evaluation scripts
