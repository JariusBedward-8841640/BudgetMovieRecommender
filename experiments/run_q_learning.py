"""Train and evaluate tabular Q-Learning agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.q_learning import QLearningAgent, QLearningConfig
from env.movie_env import BudgetMovieEnv, EnvConfig
from experiments.cli_utils import ensure_profile_args_valid, parse_profile_mix
from experiments.metrics import print_metrics, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q-Learning experiment.")
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.997)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--question-budget", type=int, default=2)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument(
        "--profile-mix",
        type=str,
        default=None,
        help="Comma-separated profiles (used when --profile is not set).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-out", type=str, default="results/models/q_learning.pkl")
    parser.add_argument("--results-out", type=str, default="results/q_learning_metrics.json")
    return parser.parse_args()


def make_tabular_env_factory(args: argparse.Namespace):
    profile_mix = parse_profile_mix(args.profile_mix)
    def _factory() -> BudgetMovieEnv:
        return BudgetMovieEnv(
            EnvConfig(
                max_steps=args.max_steps,
                question_budget=args.question_budget,
                observation_mode="tabular",
                user_profile=args.profile,
                profile_mix=profile_mix,
                seed=args.seed,
            )
        )

    return _factory


def main() -> None:
    args = parse_args()
    ensure_profile_args_valid(args.profile, args.profile_mix)
    env_factory = make_tabular_env_factory(args)
    sample_env = env_factory()
    state_size = int(sample_env.observation_space.n)
    action_size = int(sample_env.action_space.n)

    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        config=QLearningConfig(
            episodes=args.train_episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed,
        ),
    )

    train_metrics = agent.train(env_factory=env_factory, episodes=args.train_episodes)
    eval_metrics = agent.evaluate(env_factory=env_factory, episodes=args.eval_episodes, seed=args.seed)

    model_out = Path(args.model_out)
    agent.save(model_out)

    payload = {
        "algorithm": "q_learning",
        "kind": "rl",
        "config": {
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "max_steps": args.max_steps,
            "question_budget": args.question_budget,
            "profile": args.profile,
            "profile_mix": args.profile_mix,
            "seed": args.seed,
        },
        "training": train_metrics,
        "metrics": eval_metrics,
        "artifacts": {"model": str(model_out)},
    }
    save_json(args.results_out, payload)

    print("Q-Learning complete.")
    print(f"Model saved to: {model_out}")
    print(f"Metrics saved to: {args.results_out}")
    print_metrics(eval_metrics)


if __name__ == "__main__":
    main()
