"""Shared rollout and metrics helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from env.movie_env import BudgetMovieEnv


PolicyFn = Callable[[Any, dict[str, Any], BudgetMovieEnv], int]


@dataclass
class EpisodeStats:
    cumulative_reward: float
    accepted: int
    skipped: int
    abandoned: int
    questions_asked: int
    question_attempts: int
    invalid_question_attempts: int
    steps: int


def run_episode(env: BudgetMovieEnv, policy_fn: PolicyFn, seed: int | None = None) -> EpisodeStats:
    observation, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    accepted = 0
    skipped = 0
    abandoned = 0
    questions_asked = 0
    question_attempts = 0
    invalid_question_attempts = 0
    steps = 0

    while not done:
        action = policy_fn(observation, info, env)
        observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        total_reward += reward
        accepted += int(info.get("recommendation_accepted", False))
        skipped += int(info.get("recommendation_skipped", False))
        abandoned += int(info.get("abandoned", False))
        questions_asked += int(info.get("question_consumed_budget", info.get("asked_question", False)))
        question_attempts += int(info.get("question_attempted", False))
        invalid_question_attempts += int(info.get("invalid_question_action", False))

    return EpisodeStats(
        cumulative_reward=total_reward,
        accepted=accepted,
        skipped=skipped,
        abandoned=abandoned,
        questions_asked=questions_asked,
        question_attempts=question_attempts,
        invalid_question_attempts=invalid_question_attempts,
        steps=steps,
    )


def aggregate_stats(episodes: list[EpisodeStats]) -> dict[str, float]:
    if not episodes:
        raise ValueError("No episodes provided.")

    total_recommendations = sum(ep.accepted + ep.skipped for ep in episodes)
    total_questions = sum(ep.questions_asked for ep in episodes)
    total_question_attempts = sum(ep.question_attempts for ep in episodes)
    total_invalid_question_attempts = sum(ep.invalid_question_attempts for ep in episodes)
    total_accepts = sum(ep.accepted for ep in episodes)
    total_skips = sum(ep.skipped for ep in episodes)
    total_abandons = sum(ep.abandoned for ep in episodes)

    return {
        "num_episodes": float(len(episodes)),
        "avg_cumulative_reward": mean(ep.cumulative_reward for ep in episodes),
        "acceptance_rate": (
            (total_accepts / total_recommendations) if total_recommendations > 0 else 0.0
        ),
        "skip_rate": ((total_skips / total_recommendations) if total_recommendations > 0 else 0.0),
        "abandonment_rate": (total_abandons / len(episodes)),
        "avg_session_length": mean(ep.steps for ep in episodes),
        "avg_questions_asked": mean(ep.questions_asked for ep in episodes),
        "avg_question_attempts": mean(ep.question_attempts for ep in episodes),
        "avg_invalid_question_attempts": mean(ep.invalid_question_attempts for ep in episodes),
        "question_efficiency": ((total_accepts / total_questions) if total_questions > 0 else 0.0),
        "invalid_question_rate": (
            (total_invalid_question_attempts / total_question_attempts)
            if total_question_attempts > 0
            else 0.0
        ),
    }


def evaluate_policy(
    env_factory: Callable[[], BudgetMovieEnv],
    policy_fn: PolicyFn,
    episodes: int,
    seed: int | None = None,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    eps: list[EpisodeStats] = []
    for i in range(episodes):
        env = env_factory()
        episode_seed = None if seed is None else seed + i
        eps.append(run_episode(env, policy_fn, seed=episode_seed))

    result = aggregate_stats(eps)
    result["episodes"] = [asdict(ep) for ep in eps]
    return result


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def print_metrics(metrics: dict[str, Any], title: str | None = None) -> None:
    """Print metrics in a consistent, readable order."""
    if title:
        print(title)
    preferred_order = [
        "num_episodes",
        "avg_cumulative_reward",
        "acceptance_rate",
        "skip_rate",
        "abandonment_rate",
        "avg_session_length",
        "avg_questions_asked",
        "avg_question_attempts",
        "avg_invalid_question_attempts",
        "question_efficiency",
        "invalid_question_rate",
    ]
    keys = [k for k in preferred_order if k in metrics] + [
        k for k in metrics.keys() if k not in preferred_order
    ]
    for key in keys:
        print(f"- {key}: {metrics[key]}")
