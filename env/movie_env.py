"""Gymnasium environment for budgeted interactive movie recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reward_logic import RewardConfig, calculate_reward
from .state_encoder import EncoderConfig, encode_tabular, encode_vector, tabular_state_space_size
from .user_simulator import GENRES, PROFILE_NAMES, SimulatedUser, sample_profile_name

ObservationMode = Literal["tabular", "vector"]

ACTION_MAPPING = {
    0: "ask_familiar_vs_exploratory",
    1: "ask_serious_vs_light",
    2: "ask_fast_vs_calm",
    3: "recommend_action",
    4: "recommend_comedy",
    5: "recommend_drama",
    6: "recommend_scifi",
    7: "recommend_documentary",
}

QUESTION_BELIEF_WEIGHTS = {
    0: np.array([0.28, 0.16, 0.16, -0.22, -0.22], dtype=np.float32),
    1: np.array([-0.20, -0.20, 0.30, -0.06, 0.24], dtype=np.float32),
    2: np.array([0.24, -0.08, -0.24, 0.24, -0.16], dtype=np.float32),
}


@dataclass(frozen=True)
class EnvConfig:
    """Runtime configuration for the budgeted recommendation environment."""

    max_steps: int = 8
    question_budget: int = 2
    observation_mode: ObservationMode = "vector"
    reward_config: RewardConfig = RewardConfig()
    user_profile: str | None = None
    profile_mix: list[str] | None = None
    seed: int | None = None


class BudgetMovieEnv(gym.Env):
    """Custom Gymnasium environment with ask-vs-recommend decisions."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()

        if self.config.observation_mode not in ("tabular", "vector"):
            raise ValueError("observation_mode must be 'tabular' or 'vector'.")
        if self.config.user_profile and self.config.user_profile not in PROFILE_NAMES:
            raise ValueError(f"Unknown user profile '{self.config.user_profile}'.")
        if self.config.profile_mix:
            for profile in self.config.profile_mix:
                if profile not in PROFILE_NAMES:
                    raise ValueError(f"Unknown profile in mix '{profile}'.")

        self._rng = np.random.default_rng(self.config.seed)
        self._encoder_config = EncoderConfig(
            max_steps=self.config.max_steps,
            question_budget=self.config.question_budget,
            num_genres=len(GENRES),
        )

        self.action_space = spaces.Discrete(8)
        if self.config.observation_mode == "tabular":
            self.observation_space = spaces.Discrete(tabular_state_space_size(self._encoder_config))
        else:
            # Vector layout: 3 + 5 + 1 + 3 + 4 + 1 = 17
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(17,),
                dtype=np.float32,
            )

        self.current_user: SimulatedUser | None = None
        self.current_profile_name: str = ""
        self.step_index = 0
        self.remaining_questions = self.config.question_budget
        self.belief_scores = np.zeros(len(GENRES), dtype=np.float32)
        self.uncertainty = 1.0
        self.last_action_type = "none"
        self.recent_outcome = "none"
        self.last_recommended_genre: int | None = None
        self.repetition_level = 0
        self.total_questions_asked = 0
        self.done = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_index = 0
        self.remaining_questions = self.config.question_budget
        self.belief_scores = np.zeros(len(GENRES), dtype=np.float32)
        self.uncertainty = 1.0
        self.last_action_type = "none"
        self.recent_outcome = "none"
        self.last_recommended_genre = None
        self.repetition_level = 0
        self.total_questions_asked = 0
        self.done = False

        if self.config.user_profile:
            self.current_profile_name = self.config.user_profile
        else:
            self.current_profile_name = sample_profile_name(self._rng, self.config.profile_mix)
        self.current_user = SimulatedUser(self.current_profile_name, self._rng)

        observation = self._observation()
        info = self._step_info(
            action=-1,
            reward=0.0,
            outcome="none",
            abandoned=False,
            accepted=False,
            skipped=False,
            asked=False,
            question_attempted=False,
            question_consumed_budget=False,
            invalid_question_action=False,
        )
        return observation, info

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be in [0, 7], got {action}.")
        if self.current_user is None:
            raise RuntimeError("Environment not initialized. Call reset().")

        asked = False
        question_attempted = False
        question_consumed_budget = False
        invalid_question_action = False
        accepted = False
        skipped = False
        abandoned = False
        outcome = "none"

        if action <= 2:
            question_attempted = True
            self.last_action_type = "question"
            if self.remaining_questions > 0:
                asked = True
                question_consumed_budget = True
                self.remaining_questions -= 1
                self.total_questions_asked += 1
                answer, left = self.current_user.ask_question(action)
                outcome = "question"
                self.belief_scores = np.clip(
                    self.belief_scores + float(answer) * QUESTION_BELIEF_WEIGHTS[action],
                    -1.0,
                    1.0,
                )
                if left:
                    abandoned = True
                    outcome = "leave"
            else:
                # Invalid question when budget is exhausted.
                outcome = "invalid_question"
                invalid_question_action = True
                self.current_user.engagement = float(
                    np.clip(self.current_user.engagement - 0.06, 0.0, 1.0)
                )

        else:
            self.last_action_type = "recommend"
            genre_index = action - 3

            if self.last_recommended_genre == genre_index:
                self.repetition_level += 1
            else:
                self.repetition_level = 1
            self.last_recommended_genre = genre_index

            outcome = self.current_user.recommend(genre_index, self.repetition_level)
            if outcome == "accept":
                accepted = True
                self.belief_scores[genre_index] = float(
                    np.clip(self.belief_scores[genre_index] + 0.25, -1.0, 1.0)
                )
            elif outcome == "skip":
                skipped = True
                self.belief_scores[genre_index] = float(
                    np.clip(self.belief_scores[genre_index] - 0.12, -1.0, 1.0)
                )
            else:
                abandoned = True
                self.belief_scores[genre_index] = float(
                    np.clip(self.belief_scores[genre_index] - 0.15, -1.0, 1.0)
                )

        self.step_index += 1
        self.uncertainty = self._compute_uncertainty()
        self.recent_outcome = outcome if outcome in ("accept", "skip", "leave") else "none"

        truncated = self.step_index >= self.config.max_steps
        terminated = abandoned
        self.done = bool(terminated or truncated)

        reward = calculate_reward(
            action_type=self.last_action_type,
            outcome=outcome,
            repetition_level=self.repetition_level,
            done=self.done,
            config=self.config.reward_config,
        )

        observation = self._observation()
        info = self._step_info(
            action=action,
            reward=reward,
            outcome=outcome,
            abandoned=abandoned,
            accepted=accepted,
            skipped=skipped,
            asked=asked,
            question_attempted=question_attempted,
            question_consumed_budget=question_consumed_budget,
            invalid_question_action=invalid_question_action,
        )
        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self.current_user is None:
            print("Environment not started. Call reset().")
            return
        print(
            "Step:",
            self.step_index,
            "| Profile:",
            self.current_profile_name,
            "| Engagement:",
            f"{self.current_user.engagement:.2f}",
            "| RemainingQ:",
            self.remaining_questions,
            "| Belief:",
            np.round(self.belief_scores, 2).tolist(),
            "| Uncertainty:",
            f"{self.uncertainty:.2f}",
        )

    def _observation(self):
        if self.current_user is None:
            raise RuntimeError("No current user. Call reset().")
        kwargs = dict(
            step_index=self.step_index,
            max_steps=self.config.max_steps,
            remaining_budget=self.remaining_questions,
            question_budget=self.config.question_budget,
            engagement=self.current_user.engagement,
            belief_scores=self.belief_scores,
            uncertainty=self.uncertainty,
            last_action_type=self.last_action_type,
            recent_outcome=self.recent_outcome,
            repetition_level=self.repetition_level,
        )
        if self.config.observation_mode == "tabular":
            return encode_tabular(**kwargs)
        return encode_vector(**kwargs)

    def _compute_uncertainty(self) -> float:
        # Low magnitude belief implies higher uncertainty.
        certainty = float(np.mean(np.abs(self.belief_scores)))
        return float(np.clip(1.0 - certainty, 0.0, 1.0))

    def _step_info(
        self,
        *,
        action: int,
        reward: float,
        outcome: str,
        abandoned: bool,
        accepted: bool,
        skipped: bool,
        asked: bool,
        question_attempted: bool,
        question_consumed_budget: bool,
        invalid_question_action: bool,
    ) -> dict[str, Any]:
        engagement = self.current_user.engagement if self.current_user else 0.0
        return {
            "action": action,
            "action_name": ACTION_MAPPING.get(action, "reset"),
            "reward": float(reward),
            "profile": self.current_profile_name,
            "step_index": self.step_index,
            "remaining_question_budget": self.remaining_questions,
            # Backward-compatible key: now strictly means budget-consuming valid question.
            "asked_question": asked,
            "question_attempted": question_attempted,
            "question_consumed_budget": question_consumed_budget,
            "invalid_question_action": invalid_question_action,
            "recommendation_accepted": accepted,
            "recommendation_skipped": skipped,
            "abandoned": abandoned,
            "outcome": outcome,
            "engagement": float(engagement),
            "repetition_level": self.repetition_level,
            "total_questions_asked": self.total_questions_asked,
        }
