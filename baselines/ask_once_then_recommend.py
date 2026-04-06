"""Baseline: ask one question then always recommend Action."""

from __future__ import annotations

from baselines.common import make_env_factory, parse_common_args, recommend_by_belief, run_and_report


def policy_fn(observation, info, _env) -> int:
    if info.get("total_questions_asked", 0) < 1 and info.get("remaining_question_budget", 0) > 0:
        return 1
    return recommend_by_belief(observation)


def main() -> None:
    args = parse_common_args("Ask once then recommend baseline.")
    env_factory = make_env_factory(args)
    run_and_report(
        baseline_name="ask_once_then_recommend",
        policy_fn=policy_fn,
        episodes=args.episodes,
        env_factory=env_factory,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
