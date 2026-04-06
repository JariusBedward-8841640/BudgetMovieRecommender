"""Run all baseline policies with a consistent config."""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.always_ask import policy_fn as always_ask_policy
from baselines.always_recommend import policy_fn as always_recommend_policy
from baselines.ask_once_then_recommend import policy_fn as ask_once_policy
from baselines.random_policy import _make_random_policy
from env.movie_env import BudgetMovieEnv, EnvConfig
from experiments.cli_utils import ensure_profile_args_valid, parse_profile_mix
from experiments.metrics import evaluate_policy, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all baseline experiments.")
    parser.add_argument("--episodes", type=int, default=300)
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
    parser.add_argument("--results-dir", type=str, default="results/baselines")
    return parser.parse_args()


def make_env_factory(args: argparse.Namespace):
    profile_mix = parse_profile_mix(args.profile_mix)
    def _factory() -> BudgetMovieEnv:
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

    return _factory


def main() -> None:
    args = parse_args()
    ensure_profile_args_valid(args.profile, args.profile_mix)
    env_factory = make_env_factory(args)
    random_policy = _make_random_policy(args.seed)

    policies = {
        "always_recommend": always_recommend_policy,
        "always_ask": always_ask_policy,
        "ask_once_then_recommend": ask_once_policy,
        "random_policy": random_policy,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name, policy in policies.items():
        result = evaluate_policy(
            env_factory=env_factory,
            policy_fn=policy,
            episodes=args.episodes,
            seed=args.seed,
        )
        payload = {
            "algorithm": name,
            "kind": "baseline",
            "config": {
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "question_budget": args.question_budget,
                "profile": args.profile,
                "seed": args.seed,
            },
            "metrics": {k: v for k, v in result.items() if k != "episodes"},
            "episode_details": result["episodes"],
        }
        save_json(results_dir / f"{name}.json", payload)
        summary[name] = payload["metrics"]
        print(f"Completed baseline: {name}")

    save_json(results_dir / "summary.json", {"baselines": summary})
    print(f"Saved baseline outputs to: {results_dir}")


if __name__ == "__main__":
    main()
