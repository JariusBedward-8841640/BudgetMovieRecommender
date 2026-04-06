"""Session replay helpers for the dashboard product demo view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import numpy as np

from agents.q_learning import QLearningAgent
from baselines.always_ask import policy_fn as always_ask_policy
from baselines.always_recommend import policy_fn as always_recommend_policy
from baselines.ask_once_then_recommend import policy_fn as ask_once_policy
from baselines.random_policy import _make_random_policy
from env.movie_env import BudgetMovieEnv, EnvConfig
from env.user_simulator import PROFILE_NAMES


try:
    from stable_baselines3 import DQN
except ImportError:  # pragma: no cover - optional dependency at runtime
    DQN = None  # type: ignore[assignment]


@dataclass
class ReplayResult:
    steps: list[dict[str, Any]]
    summary: dict[str, Any]
    warning: str | None = None


def available_profiles() -> list[str]:
    return list(PROFILE_NAMES)


def build_policy_map(model_paths: dict[str, str]) -> dict[str, str]:
    policies = {
        "always_recommend": "Always Recommend",
        "always_ask": "Always Ask",
        "ask_once_then_recommend": "Ask Once Then Recommend",
        "random_policy": "Random Policy",
    }
    if "q_learning" in model_paths:
        policies["q_learning"] = "Q-Learning (trained model)"
    if "dqn" in model_paths:
        policies["dqn"] = "DQN (trained model)"
    return policies


def _policy_callable(
    policy_key: str,
    seed: int,
    model_paths: dict[str, str],
) -> tuple[Callable[[Any, dict[str, Any], BudgetMovieEnv], int] | None, str]:
    if policy_key == "always_recommend":
        return always_recommend_policy, "vector"
    if policy_key == "always_ask":
        return always_ask_policy, "vector"
    if policy_key == "ask_once_then_recommend":
        return ask_once_policy, "vector"
    if policy_key == "random_policy":
        return _make_random_policy(seed), "vector"
    if policy_key == "q_learning":
        model_path = model_paths.get("q_learning")
        if not model_path or not Path(model_path).exists():
            return None, "tabular"
        agent = QLearningAgent.load(model_path)

        def _q_policy(observation, _info, _env):
            return int(np.argmax(agent.q_table[int(observation)]))

        return _q_policy, "tabular"
    if policy_key == "dqn":
        if DQN is None:
            return None, "vector"
        model_path = model_paths.get("dqn")
        if not model_path or not Path(model_path).exists():
            return None, "vector"
        model = DQN.load(model_path)

        def _dqn_policy(observation, _info, _env):
            action, _ = model.predict(observation, deterministic=True)
            return int(action)

        return _dqn_policy, "vector"
    return None, "vector"


def replay_session(
    *,
    policy_key: str,
    profile: str,
    question_budget: int,
    max_steps: int,
    seed: int,
    model_paths: dict[str, str],
) -> ReplayResult:
    policy, observation_mode = _policy_callable(policy_key, seed, model_paths)
    if policy is None:
        return ReplayResult(
            steps=[],
            summary={},
            warning=f"Policy '{policy_key}' is unavailable. Missing model artifact or dependency.",
        )

    env = BudgetMovieEnv(
        EnvConfig(
            max_steps=max_steps,
            question_budget=question_budget,
            observation_mode=observation_mode,  # tabular for q-learning, vector otherwise
            user_profile=profile,
            seed=seed,
        )
    )
    observation, info = env.reset(seed=seed)
    done = False
    steps: list[dict[str, Any]] = []
    total_reward = 0.0
    accepts = 0
    skips = 0
    abandons = 0

    while not done:
        action = policy(observation, info, env)
        observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += reward
        accepts += int(info.get("recommendation_accepted", False))
        skips += int(info.get("recommendation_skipped", False))
        abandons += int(info.get("abandoned", False))
        steps.append(
            {
                "step": int(info.get("step_index", len(steps) + 1)),
                "action": info.get("action_name"),
                "outcome": info.get("outcome"),
                "reward": float(reward),
                "engagement": float(info.get("engagement", 0.0)),
                "remaining_question_budget": int(info.get("remaining_question_budget", 0)),
                "question_attempted": bool(info.get("question_attempted", False)),
                "question_consumed_budget": bool(info.get("question_consumed_budget", False)),
                "invalid_question_action": bool(info.get("invalid_question_action", False)),
            }
        )

    summary = {
        "accepted_recommendations": accepts,
        "skipped_recommendations": skips,
        "abandoned": bool(abandons > 0),
        "total_reward": round(total_reward, 4),
        "questions_used": int(info.get("total_questions_asked", 0)),
        "session_steps": len(steps),
    }
    return ReplayResult(steps=steps, summary=summary)
