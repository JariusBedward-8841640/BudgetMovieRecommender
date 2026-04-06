"""Run a reproducible comparison sweep across budgets and profile settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.q_learning import QLearningAgent, QLearningConfig
from agents.train_dqn import DQNTrainConfig, evaluate_dqn, save_dqn_model, train_dqn
from baselines.always_ask import policy_fn as always_ask_policy
from baselines.always_recommend import policy_fn as always_recommend_policy
from baselines.ask_once_then_recommend import policy_fn as ask_once_policy
from baselines.random_policy import _make_random_policy
from env.movie_env import BudgetMovieEnv, EnvConfig
from env.user_simulator import PROFILE_NAMES
from experiments.compare_results import run_comparison
from experiments.metrics import evaluate_policy, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full comparison sweep.")
    parser.add_argument("--budgets", type=str, default="0,1,2,3")
    parser.add_argument(
        "--profile-settings",
        type=str,
        default="mixed,action_focused,balanced_viewer,novelty_seeking,question_sensitive",
        help="Comma-separated values, where 'mixed' means random sampling over all profiles.",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--baseline-episodes", type=int, default=200)
    parser.add_argument("--q-train-episodes", type=int, default=2000)
    parser.add_argument("--q-eval-episodes", type=int, default=200)
    parser.add_argument("--dqn-total-timesteps", type=int, default=20_000)
    parser.add_argument("--dqn-eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--results-root", type=str, default="results/sweeps")
    return parser.parse_args()


def _parse_budgets(raw: str) -> list[int]:
    budgets = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not budgets:
        raise ValueError("No budgets provided.")
    return budgets


def _parse_profile_setting(setting: str) -> tuple[str | None, list[str] | None]:
    setting = setting.strip()
    if setting == "mixed":
        return None, list(PROFILE_NAMES)
    if setting not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile setting '{setting}'.")
    return setting, None


def _env_factory(max_steps: int, budget: int, profile: str | None, profile_mix: list[str] | None, seed: int):
    def _factory() -> BudgetMovieEnv:
        return BudgetMovieEnv(
            EnvConfig(
                max_steps=max_steps,
                question_budget=budget,
                observation_mode="vector",
                user_profile=profile,
                profile_mix=profile_mix,
                seed=seed,
            )
        )

    return _factory


def _tabular_env_factory(
    max_steps: int, budget: int, profile: str | None, profile_mix: list[str] | None, seed: int
):
    def _factory() -> BudgetMovieEnv:
        return BudgetMovieEnv(
            EnvConfig(
                max_steps=max_steps,
                question_budget=budget,
                observation_mode="tabular",
                user_profile=profile,
                profile_mix=profile_mix,
                seed=seed,
            )
        )

    return _factory


def main() -> None:
    args = parse_args()
    budgets = _parse_budgets(args.budgets)
    profile_settings = [item.strip() for item in args.profile_settings.split(",") if item.strip()]
    policies = {
        "always_recommend": always_recommend_policy,
        "always_ask": always_ask_policy,
        "ask_once_then_recommend": ask_once_policy,
    }

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    for budget in budgets:
        for setting in profile_settings:
            profile, profile_mix = _parse_profile_setting(setting)
            profile_label = setting

            run_dir = results_root / f"budget_{budget}" / profile_label
            run_dir.mkdir(parents=True, exist_ok=True)

            vector_factory = _env_factory(
                args.max_steps,
                budget,
                profile,
                profile_mix,
                args.seed,
            )
            tabular_factory = _tabular_env_factory(
                args.max_steps,
                budget,
                profile,
                profile_mix,
                args.seed,
            )

            random_policy = _make_random_policy(args.seed)
            all_policies = dict(policies)
            all_policies["random_policy"] = random_policy

            for name, policy_fn in all_policies.items():
                result = evaluate_policy(
                    env_factory=vector_factory,
                    policy_fn=policy_fn,
                    episodes=args.baseline_episodes,
                    seed=args.seed,
                )
                payload = {
                    "algorithm": name,
                    "kind": "baseline",
                    "config": {
                        "question_budget": budget,
                        "profile": profile,
                        "profile_mix": ",".join(profile_mix) if profile_mix else None,
                        "episodes": args.baseline_episodes,
                        "seed": args.seed,
                    },
                    "metrics": {k: v for k, v in result.items() if k != "episodes"},
                }
                save_json(run_dir / f"{name}.json", payload)

            q_env = tabular_factory()
            q_agent = QLearningAgent(
                state_size=int(q_env.observation_space.n),
                action_size=int(q_env.action_space.n),
                config=QLearningConfig(episodes=args.q_train_episodes, seed=args.seed),
            )
            q_agent.train(tabular_factory, episodes=args.q_train_episodes)
            q_eval = q_agent.evaluate(tabular_factory, episodes=args.q_eval_episodes, seed=args.seed)
            q_model_path = run_dir / "q_learning.pkl"
            q_agent.save(q_model_path)
            save_json(
                run_dir / "q_learning.json",
                {
                    "algorithm": "q_learning",
                    "kind": "rl",
                    "config": {
                        "question_budget": budget,
                        "profile": profile,
                        "profile_mix": ",".join(profile_mix) if profile_mix else None,
                        "train_episodes": args.q_train_episodes,
                        "eval_episodes": args.q_eval_episodes,
                        "seed": args.seed,
                    },
                    "metrics": q_eval,
                    "artifacts": {"model": str(q_model_path)},
                },
            )

            dqn_train_env = vector_factory()
            dqn_model = train_dqn(
                dqn_train_env,
                DQNTrainConfig(total_timesteps=args.dqn_total_timesteps, seed=args.seed),
            )
            dqn_eval = evaluate_dqn(
                dqn_model,
                env_factory=vector_factory,
                episodes=args.dqn_eval_episodes,
                seed=args.seed,
            )
            dqn_model_path = run_dir / "dqn_model"
            save_dqn_model(dqn_model, dqn_model_path)
            save_json(
                run_dir / "dqn.json",
                {
                    "algorithm": "dqn",
                    "kind": "rl",
                    "config": {
                        "question_budget": budget,
                        "profile": profile,
                        "profile_mix": ",".join(profile_mix) if profile_mix else None,
                        "total_timesteps": args.dqn_total_timesteps,
                        "eval_episodes": args.dqn_eval_episodes,
                        "seed": args.seed,
                    },
                    "metrics": dqn_eval,
                    "artifacts": {"model": str(dqn_model_path)},
                },
            )

            print(f"Completed sweep: budget={budget}, profile_setting={profile_label}")

    run_comparison(
        results_root=results_root,
        output_json=results_root / "comparison_summary.json",
        output_csv=results_root / "comparison_summary.csv",
    )


if __name__ == "__main__":
    main()
