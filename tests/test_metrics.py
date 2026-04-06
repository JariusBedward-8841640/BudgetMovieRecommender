from env.movie_env import BudgetMovieEnv, EnvConfig
from experiments.metrics import run_episode


def _always_question_policy(_observation, _info, _env) -> int:
    return 0


def test_metrics_count_only_budget_consuming_questions():
    env = BudgetMovieEnv(
        EnvConfig(
            observation_mode="vector",
            question_budget=2,
            max_steps=6,
            user_profile="balanced_viewer",
            seed=33,
        )
    )
    stats = run_episode(env, _always_question_policy, seed=33)
    assert stats.questions_asked <= 2
    assert stats.question_attempts >= stats.questions_asked
    assert stats.invalid_question_attempts >= 0
