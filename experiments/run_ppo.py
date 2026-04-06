"""Optional PPO experiment runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.train_ppo import PPOTrainConfig, evaluate_ppo, save_ppo_model, train_ppo
from env.movie_env import BudgetMovieEnv, EnvConfig
from experiments.metrics import print_metrics, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional PPO experiment.")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--question-budget", type=int, default=2)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-out", type=str, default="results/models/ppo_model")
    parser.add_argument("--results-out", type=str, default="results/ppo_metrics.json")
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> BudgetMovieEnv:
    return BudgetMovieEnv(
        EnvConfig(
            max_steps=args.max_steps,
            question_budget=args.question_budget,
            observation_mode="vector",
            user_profile=args.profile,
            seed=args.seed,
        )
    )


def make_env_factory(args: argparse.Namespace):
    def _factory() -> BudgetMovieEnv:
        return make_env(args)

    return _factory


def main() -> None:
    args = parse_args()
    env = make_env(args)
    env_factory = make_env_factory(args)

    config = PPOTrainConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        seed=args.seed,
    )

    model = train_ppo(env, config)
    metrics = evaluate_ppo(model, env_factory=env_factory, episodes=args.eval_episodes, seed=args.seed)
    save_ppo_model(model, Path(args.model_out))

    payload = {
        "algorithm": "ppo",
        "kind": "rl_optional",
        "config": vars(args),
        "metrics": metrics,
        "artifacts": {"model": args.model_out},
    }
    save_json(args.results_out, payload)

    print("PPO complete.")
    print(f"Model saved to: {args.model_out}")
    print(f"Metrics saved to: {args.results_out}")
    print_metrics(metrics)


if __name__ == "__main__":
    main()
