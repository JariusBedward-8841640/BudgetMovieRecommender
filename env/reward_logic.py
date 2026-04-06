"""Reward logic for the budgeted interactive movie recommendation environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Configurable reward components."""

    accepted_reward: float = 2.2
    continuation_bonus: float = 0.2
    skipped_penalty: float = -0.8
    abandonment_penalty: float = -3.0
    question_cost: float = -0.08
    invalid_question_penalty: float = -0.35
    repetition_penalty: float = -0.25


def calculate_reward(
    *,
    action_type: str,
    outcome: str,
    repetition_level: int,
    done: bool,
    config: RewardConfig,
) -> float:
    """Calculate step reward from action/outcome context."""
    reward = 0.0

    if action_type == "question":
        reward += config.question_cost
        if outcome == "invalid_question":
            reward += config.invalid_question_penalty
        return reward

    if action_type == "recommend":
        if outcome == "accept":
            reward += config.accepted_reward
            if not done:
                reward += config.continuation_bonus
        elif outcome == "skip":
            reward += config.skipped_penalty
        elif outcome == "leave":
            reward += config.abandonment_penalty

        repeated_extra = max(0, repetition_level - 1)
        reward += repeated_extra * config.repetition_penalty
        return reward

    raise ValueError(f"Unknown action_type '{action_type}'.")
