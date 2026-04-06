"""Tabular Q-Learning implementation for BudgetMovieEnv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from env.movie_env import BudgetMovieEnv
from experiments.metrics import aggregate_stats, run_episode


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 2000
    alpha: float = 0.15
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997
    seed: int = 7


class QLearningAgent:
    """Minimal, reproducible epsilon-greedy Q-Learning agent."""

    def __init__(self, state_size: int, action_size: int, config: QLearningConfig | None = None):
        self.config = config or QLearningConfig()
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)
        self.rng = np.random.default_rng(self.config.seed)
        self.epsilon = self.config.epsilon_start

    def select_action(self, state: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_size))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        td_target = reward + (0.0 if done else self.config.gamma * np.max(self.q_table[next_state]))
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.config.alpha * td_error

    def train(self, env_factory, episodes: int | None = None) -> dict[str, float]:
        total_episodes = episodes or self.config.episodes
        reward_history: list[float] = []

        for episode_idx in range(total_episodes):
            env: BudgetMovieEnv = env_factory()
            state, _ = env.reset(seed=self.config.seed + episode_idx)
            done = False
            episode_reward = 0.0

            while not done:
                action = self.select_action(int(state), explore=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                self.update(int(state), action, reward, int(next_state), done)
                state = int(next_state)
                episode_reward += reward

            reward_history.append(episode_reward)
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        return {
            "train_episodes": float(total_episodes),
            "train_avg_reward_last_100": float(np.mean(reward_history[-100:])),
            "final_epsilon": float(self.epsilon),
        }

    def evaluate(self, env_factory, episodes: int = 200, seed: int | None = None) -> dict:
        eval_seed = self.config.seed if seed is None else seed
        episodes_stats = []

        def greedy_policy(_obs, _info, env: BudgetMovieEnv) -> int:
            obs = _obs
            if not isinstance(obs, (int, np.integer)):
                raise TypeError("Q-Learning evaluation requires tabular observations.")
            return int(np.argmax(self.q_table[int(obs)]))

        for i in range(episodes):
            env = env_factory()
            episodes_stats.append(run_episode(env, greedy_policy, seed=eval_seed + i))

        return aggregate_stats(episodes_stats)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "epsilon": self.epsilon,
            "q_table": self.q_table,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        agent = cls(
            state_size=payload["state_size"],
            action_size=payload["action_size"],
            config=payload["config"],
        )
        agent.epsilon = payload["epsilon"]
        agent.q_table = payload["q_table"]
        return agent
