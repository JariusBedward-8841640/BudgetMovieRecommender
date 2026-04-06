"""Baseline: uniformly random over full action space."""

from __future__ import annotations

import numpy as np

from baselines.common import make_env_factory, parse_common_args, run_and_report


def _make_random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def policy_fn(_observation, _info, _env) -> int:
        return int(rng.integers(0, 8))

    return policy_fn


def main() -> None:
    args = parse_common_args("Random policy baseline.")
    env_factory = make_env_factory(args)
    run_and_report(
        baseline_name="random_policy",
        policy_fn=_make_random_policy(args.seed),
        episodes=args.episodes,
        env_factory=env_factory,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
