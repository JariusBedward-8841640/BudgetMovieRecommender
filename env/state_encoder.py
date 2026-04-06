"""State encoders for tabular and vector representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncoderConfig:
    max_steps: int
    question_budget: int
    num_genres: int = 5


def _bin_engagement(value: float) -> int:
    if value < 0.34:
        return 0
    if value < 0.67:
        return 1
    return 2


def _bin_uncertainty(value: float) -> int:
    if value < 0.34:
        return 0
    if value < 0.67:
        return 1
    return 2


def _bin_repetition(level: int) -> int:
    if level <= 1:
        return 0
    if level == 2:
        return 1
    return 2


def tabular_state_space_size(config: EncoderConfig) -> int:
    """Number of tabular states under this discretization."""
    step_bins = config.max_steps
    budget_bins = config.question_budget + 1
    engagement_bins = 3
    top_genre_bins = config.num_genres
    uncertainty_bins = 3
    last_action_bins = 3  # none/question/recommend
    outcome_bins = 4  # none/accept/skip/leave
    repetition_bins = 3
    return (
        step_bins
        * budget_bins
        * engagement_bins
        * top_genre_bins
        * uncertainty_bins
        * last_action_bins
        * outcome_bins
        * repetition_bins
    )


def encode_tabular(
    *,
    step_index: int,
    max_steps: int,
    remaining_budget: int,
    question_budget: int,
    engagement: float,
    belief_scores: np.ndarray,
    uncertainty: float,
    last_action_type: str,
    recent_outcome: str,
    repetition_level: int,
) -> int:
    """Encode environment context into one discrete state id."""
    step_bin = int(np.clip(step_index, 0, max_steps - 1))
    budget_bin = int(np.clip(remaining_budget, 0, question_budget))
    engagement_bin = _bin_engagement(float(engagement))
    top_genre_bin = int(np.argmax(belief_scores))
    uncertainty_bin = _bin_uncertainty(float(uncertainty))

    if last_action_type == "question":
        last_action_bin = 1
    elif last_action_type == "recommend":
        last_action_bin = 2
    else:
        last_action_bin = 0

    if recent_outcome == "accept":
        outcome_bin = 1
    elif recent_outcome == "skip":
        outcome_bin = 2
    elif recent_outcome == "leave":
        outcome_bin = 3
    else:
        outcome_bin = 0

    repetition_bin = _bin_repetition(repetition_level)

    values = (
        step_bin,
        budget_bin,
        engagement_bin,
        top_genre_bin,
        uncertainty_bin,
        last_action_bin,
        outcome_bin,
        repetition_bin,
    )
    radices = (
        max_steps,
        question_budget + 1,
        3,
        len(belief_scores),
        3,
        3,
        4,
        3,
    )

    state_id = 0
    multiplier = 1
    for value, radix in zip(values, radices):
        state_id += value * multiplier
        multiplier *= radix

    return int(state_id)


def encode_vector(
    *,
    step_index: int,
    max_steps: int,
    remaining_budget: int,
    question_budget: int,
    engagement: float,
    belief_scores: np.ndarray,
    uncertainty: float,
    last_action_type: str,
    recent_outcome: str,
    repetition_level: int,
) -> np.ndarray:
    """Encode environment context into a dense vector for DQN."""
    step_norm = step_index / max(1, max_steps - 1)
    budget_norm = remaining_budget / max(1, question_budget)
    repetition_norm = min(repetition_level, 4) / 4.0

    if last_action_type == "question":
        last_action_one_hot = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif last_action_type == "recommend":
        last_action_one_hot = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        last_action_one_hot = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    if recent_outcome == "accept":
        outcome_one_hot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    elif recent_outcome == "skip":
        outcome_one_hot = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    elif recent_outcome == "leave":
        outcome_one_hot = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    else:
        outcome_one_hot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    parts = [
        np.array([step_norm, budget_norm, float(engagement)], dtype=np.float32),
        belief_scores.astype(np.float32),
        np.array([float(uncertainty)], dtype=np.float32),
        last_action_one_hot,
        outcome_one_hot,
        np.array([repetition_norm], dtype=np.float32),
    ]
    return np.concatenate(parts).astype(np.float32)
