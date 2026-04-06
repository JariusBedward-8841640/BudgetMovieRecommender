from env.movie_env import BudgetMovieEnv, EnvConfig


def test_env_reset_and_step_vector():
    env = BudgetMovieEnv(EnvConfig(observation_mode="vector", seed=11))
    obs, info = env.reset(seed=11)
    assert obs.shape == (17,)
    assert info["remaining_question_budget"] == env.config.question_budget

    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "asked_question" in info
    assert obs.shape == (17,)


def test_env_random_rollout_tabular():
    env = BudgetMovieEnv(EnvConfig(observation_mode="tabular", seed=12))
    obs, _ = env.reset(seed=12)
    assert isinstance(obs, int)

    done = False
    steps = 0
    while not done and steps < 20:
        action = env.action_space.sample()
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        assert isinstance(obs, int)

    assert steps <= env.config.max_steps


def test_budget_exhaustion_marks_invalid_question():
    env = BudgetMovieEnv(EnvConfig(observation_mode="vector", question_budget=0, seed=21))
    _obs, _info = env.reset(seed=21)
    _obs, reward, terminated, truncated, info = env.step(0)
    assert info["outcome"] == "invalid_question"
    assert info["remaining_question_budget"] == 0
    assert info["question_attempted"] is True
    assert info["question_consumed_budget"] is False
    assert info["asked_question"] is False
    assert info["invalid_question_action"] is True
    assert reward < 0
    assert not terminated
    assert not truncated


def test_valid_question_updates_budget_and_count():
    env = BudgetMovieEnv(EnvConfig(observation_mode="vector", question_budget=2, seed=23))
    _obs, _info = env.reset(seed=23)
    _obs, _reward, _terminated, _truncated, info = env.step(0)
    assert info["question_attempted"] is True
    assert info["question_consumed_budget"] is True
    assert info["asked_question"] is True
    assert info["invalid_question_action"] is False
    assert info["remaining_question_budget"] == 1
    assert info["total_questions_asked"] == 1


def test_profile_mix_uses_allowed_profiles():
    env = BudgetMovieEnv(
        EnvConfig(
            observation_mode="vector",
            profile_mix=["action_focused", "balanced_viewer"],
            seed=22,
        )
    )
    _obs, info = env.reset(seed=22)
    assert info["profile"] in {"action_focused", "balanced_viewer"}
