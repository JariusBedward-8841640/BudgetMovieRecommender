"""Shared baseline helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from env.movie_env import EnvConfig, BudgetMovieEnv
from experiments.metrics import evaluate_policy, print_metrics, save_json


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--question-budget", type=int, default=2)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output json path. If omitted, only prints metrics.",
    )
    return parser.parse_args()


def make_env_factory(args: argparse.Namespace) -> Callable[[], BudgetMovieEnv]:
    def _factory() -> BudgetMovieEnv:
        return BudgetMovieEnv(
            EnvConfig(
                max_steps=args.max_steps,
                question_budget=args.question_budget,
                observation_mode="vector",
                user_profile=args.profile,
                seed=args.seed,
            )
        )

    return _factory


def run_and_report(
    *,
    baseline_name: str,
    policy_fn,
    episodes: int,
    env_factory,
    seed: int,
    output: str | None,
) -> dict:
    result = evaluate_policy(env_factory, policy_fn, episodes=episodes, seed=seed)
    payload = {
        "algorithm": baseline_name,
        "kind": "baseline",
        "config": {"episodes": episodes},
        "metrics": {k: v for k, v in result.items() if k != "episodes"},
        "episode_details": result["episodes"],
    }
    print_metrics(payload["metrics"], title=f"{baseline_name} metrics:")
    if output:
        save_json(Path(output), payload)
    return payload


def recommend_by_belief(observation) -> int:
    """Map belief estimate in vector observation to recommend action id."""
    if isinstance(observation, np.ndarray) and observation.shape[0] >= 8:
        belief = observation[3:8]
        return 3 + int(np.argmax(belief))
    return 3
