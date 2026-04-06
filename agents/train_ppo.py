"""Optional PPO training/evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from env.movie_env import BudgetMovieEnv
from experiments.metrics import aggregate_stats, run_episode

try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover
    PPO = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PPOTrainConfig:
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    seed: int = 7


def train_ppo(env: BudgetMovieEnv, config: PPOTrainConfig):
    if PPO is None:
        raise ImportError(
            "stable-baselines3 is required for PPO. Install dependencies first."
        )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        seed=config.seed,
        verbose=0,
    )
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    return model


def evaluate_ppo(model, env_factory, episodes: int = 300, seed: int = 7) -> dict[str, Any]:
    def policy_fn(observation, _info, _env):
        action, _states = model.predict(observation, deterministic=True)
        return int(action)

    eps = []
    for i in range(episodes):
        env = env_factory()
        eps.append(run_episode(env, policy_fn, seed=seed + i))
    return aggregate_stats(eps)


def save_ppo_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
