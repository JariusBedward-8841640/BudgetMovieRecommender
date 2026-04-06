"""User simulator for the budgeted interactive movie recommender."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

PROFILE_NAMES = (
    "action_focused",
    "balanced_viewer",
    "novelty_seeking",
    "question_sensitive",
)

GENRES = ("Action", "Comedy", "Drama", "Sci-Fi", "Documentary")

QuestionAnswer = Literal[-1, 1]
RecOutcome = Literal["accept", "skip", "leave"]


@dataclass(frozen=True)
class ProfileConfig:
    """Static profile traits used to instantiate a simulated user."""

    preference_scores: np.ndarray
    initial_engagement: float
    question_sensitivity: float
    repetition_sensitivity: float
    base_leave_probability: float


def _profile_library() -> Dict[str, ProfileConfig]:
    return {
        "action_focused": ProfileConfig(
            preference_scores=np.array([0.95, 0.45, 0.25, 0.65, 0.15], dtype=np.float32),
            initial_engagement=0.78,
            question_sensitivity=0.07,
            repetition_sensitivity=0.05,
            base_leave_probability=0.03,
        ),
        "balanced_viewer": ProfileConfig(
            preference_scores=np.array([0.58, 0.55, 0.57, 0.56, 0.54], dtype=np.float32),
            initial_engagement=0.72,
            question_sensitivity=0.03,
            repetition_sensitivity=0.03,
            base_leave_probability=0.02,
        ),
        "novelty_seeking": ProfileConfig(
            preference_scores=np.array([0.55, 0.50, 0.48, 0.72, 0.62], dtype=np.float32),
            initial_engagement=0.70,
            question_sensitivity=0.04,
            repetition_sensitivity=0.14,
            base_leave_probability=0.03,
        ),
        "question_sensitive": ProfileConfig(
            preference_scores=np.array([0.62, 0.52, 0.60, 0.50, 0.48], dtype=np.float32),
            initial_engagement=0.68,
            question_sensitivity=0.13,
            repetition_sensitivity=0.05,
            base_leave_probability=0.05,
        ),
    }


class SimulatedUser:
    """A stochastic but interpretable simulated movie user."""

    def __init__(self, profile_name: str, rng: np.random.Generator):
        if profile_name not in PROFILE_NAMES:
            raise ValueError(f"Unknown profile '{profile_name}'.")

        self.profile_name = profile_name
        self._config = _profile_library()[profile_name]
        self._rng = rng
        self.preference_scores = self._config.preference_scores.copy()
        self.engagement = float(self._config.initial_engagement)
        self.question_count = 0
        self.recommendation_count = 0

    @property
    def leave_probability(self) -> float:
        """Dynamic leave probability driven by engagement and friction."""
        engagement_factor = max(0.0, 0.6 - self.engagement) * 0.4
        question_factor = self.question_count * self._config.question_sensitivity * 0.08
        raw = self._config.base_leave_probability + engagement_factor + question_factor
        return float(np.clip(raw, 0.0, 0.95))

    def ask_question(self, question_id: int) -> tuple[QuestionAnswer, bool]:
        """Return a binary answer and whether the user leaves after question."""
        if question_id not in (0, 1, 2):
            raise ValueError("Question id must be one of 0, 1, or 2.")

        self.question_count += 1
        self.engagement = float(
            np.clip(self.engagement - self._config.question_sensitivity, 0.0, 1.0)
        )

        answer = self._question_answer(question_id)
        left = bool(self._rng.random() < self.leave_probability)
        return answer, left

    def recommend(self, genre_index: int, repetition_level: int) -> RecOutcome:
        """Sample recommendation outcome from hidden preferences and context."""
        if not 0 <= genre_index < len(GENRES):
            raise ValueError(f"Genre index must be in [0, {len(GENRES) - 1}].")

        self.recommendation_count += 1
        preference = float(self.preference_scores[genre_index])

        repetition_penalty = max(0, repetition_level - 1) * self._config.repetition_sensitivity
        accept_prob = np.clip(
            0.10 + 0.70 * preference + 0.20 * self.engagement - repetition_penalty,
            0.01,
            0.98,
        )

        leave_prob = np.clip(
            self.leave_probability + max(0.0, 0.45 - preference) * 0.25 + repetition_penalty * 0.6,
            0.0,
            0.98,
        )

        draw = self._rng.random()
        if draw < accept_prob:
            self.engagement = float(np.clip(self.engagement + 0.05, 0.0, 1.0))
            return "accept"

        if draw < min(0.995, accept_prob + leave_prob):
            self.engagement = float(np.clip(self.engagement - 0.08, 0.0, 1.0))
            return "leave"

        self.engagement = float(np.clip(self.engagement - 0.03, 0.0, 1.0))
        return "skip"

    def _question_answer(self, question_id: int) -> QuestionAnswer:
        """Map hidden preferences to interpretable binary question answers."""
        if question_id == 0:
            # familiar vs exploratory: mainstream(Action/Comedy/Drama) vs exploratory(Sci-Fi/Doc)
            lhs = float(np.mean(self.preference_scores[[0, 1, 2]]))
            rhs = float(np.mean(self.preference_scores[[3, 4]]))
        elif question_id == 1:
            # serious vs light: Drama/Documentary vs Action/Comedy
            lhs = float(np.mean(self.preference_scores[[2, 4]]))
            rhs = float(np.mean(self.preference_scores[[0, 1]]))
        else:
            # fast-paced vs calm: Action/Sci-Fi vs Drama/Documentary
            lhs = float(np.mean(self.preference_scores[[0, 3]]))
            rhs = float(np.mean(self.preference_scores[[2, 4]]))

        jitter = float(self._rng.normal(loc=0.0, scale=0.03))
        return 1 if lhs + jitter >= rhs else -1


def sample_profile_name(
    rng: np.random.Generator, allowed_profiles: list[str] | None = None
) -> str:
    """Sample a profile name from full set or a subset."""
    candidates = allowed_profiles if allowed_profiles else list(PROFILE_NAMES)
    for profile in candidates:
        if profile not in PROFILE_NAMES:
            raise ValueError(f"Unknown profile in mix: {profile}")
    return str(rng.choice(candidates))
