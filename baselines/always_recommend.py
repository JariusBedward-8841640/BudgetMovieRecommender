"""Baseline: always recommend (Action by default)."""

from __future__ import annotations

from baselines.common import make_env_factory, parse_common_args, run_and_report


def policy_fn(_observation, _info, _env) -> int:
    return 3  # recommend Action


def main() -> None:
    args = parse_common_args("Always recommend baseline.")
    env_factory = make_env_factory(args)
    run_and_report(
        baseline_name="always_recommend",
        policy_fn=policy_fn,
        episodes=args.episodes,
        env_factory=env_factory,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
