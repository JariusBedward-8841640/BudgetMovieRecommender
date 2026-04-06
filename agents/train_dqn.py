"""Stable Baselines3 DQN training and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from env.movie_env import BudgetMovieEnv
from experiments.metrics import aggregate_stats, run_episode

try:
    from stable_baselines3 import DQN
except ImportError:  # pragma: no cover - handled at runtime
    DQN = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DQNTrainConfig:
    total_timesteps: int = 40_000
    learning_rate: float = 1e-3
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.98
    train_freq: int = 4
    target_update_interval: int = 500
    exploration_fraction: float = 0.25
    exploration_final_eps: float = 0.05
    seed: int = 7


def train_dqn(env: BudgetMovieEnv, config: DQNTrainConfig):
    if DQN is None:
        raise ImportError(
            "stable-baselines3 is required for DQN. Install dependencies first."
        )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        train_freq=config.train_freq,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_final_eps=config.exploration_final_eps,
        verbose=0,
        seed=config.seed,
    )
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    return model


def evaluate_dqn(
    model,
    env_factory,
    episodes: int = 300,
    seed: int = 7,
) -> dict[str, Any]:
    def policy_fn(observation, _info, _env):
        action, _states = model.predict(observation, deterministic=True)
        return int(action)

    eps = []
    for i in range(episodes):
        env = env_factory()
        eps.append(run_episode(env, policy_fn, seed=seed + i))
    return aggregate_stats(eps)


def save_dqn_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
