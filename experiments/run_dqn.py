"""Train and evaluate SB3 DQN on vector observations."""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.train_dqn import DQNTrainConfig, evaluate_dqn, save_dqn_model, train_dqn
from env.movie_env import BudgetMovieEnv, EnvConfig
from experiments.cli_utils import ensure_profile_args_valid, parse_profile_mix
from experiments.metrics import print_metrics, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DQN experiment.")
    parser.add_argument("--total-timesteps", type=int, default=40_000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
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
    parser.add_argument("--model-out", type=str, default="results/models/dqn_model")
    parser.add_argument("--results-out", type=str, default="results/dqn_metrics.json")
    return parser.parse_args()


def make_vector_env(args: argparse.Namespace) -> BudgetMovieEnv:
    profile_mix = parse_profile_mix(args.profile_mix)
    return BudgetMovieEnv(
        EnvConfig(
            max_steps=args.max_steps,
            question_budget=args.question_budget,
            observation_mode="vector",
            user_profile=args.profile,
            profile_mix=profile_mix,
            seed=args.seed,
        )
    )


def make_vector_env_factory(args: argparse.Namespace):
    def _factory() -> BudgetMovieEnv:
        return make_vector_env(args)

    return _factory


def main() -> None:
    args = parse_args()
    ensure_profile_args_valid(args.profile, args.profile_mix)
    train_env = make_vector_env(args)
    env_factory = make_vector_env_factory(args)

    config = DQNTrainConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        seed=args.seed,
    )

    model = train_dqn(train_env, config)
    eval_metrics = evaluate_dqn(model, env_factory=env_factory, episodes=args.eval_episodes, seed=args.seed)

    model_out = Path(args.model_out)
    save_dqn_model(model, model_out)

    payload = {
        "algorithm": "dqn",
        "kind": "rl",
        "config": {
            "total_timesteps": args.total_timesteps,
            "eval_episodes": args.eval_episodes,
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "train_freq": args.train_freq,
            "target_update_interval": args.target_update_interval,
            "exploration_fraction": args.exploration_fraction,
            "exploration_final_eps": args.exploration_final_eps,
            "max_steps": args.max_steps,
            "question_budget": args.question_budget,
            "profile": args.profile,
            "profile_mix": args.profile_mix,
            "seed": args.seed,
        },
        "metrics": eval_metrics,
        "artifacts": {"model": str(model_out)},
    }
    save_json(args.results_out, payload)

    print("DQN complete.")
    print(f"Model saved to: {model_out}")
    print(f"Metrics saved to: {args.results_out}")
    print_metrics(eval_metrics)


if __name__ == "__main__":
    main()
