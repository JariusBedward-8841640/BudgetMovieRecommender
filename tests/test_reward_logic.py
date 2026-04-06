from env.reward_logic import RewardConfig, calculate_reward


def test_reward_accept_positive():
    cfg = RewardConfig()
    reward = calculate_reward(
        action_type="recommend",
        outcome="accept",
        repetition_level=1,
        done=False,
        config=cfg,
    )
    assert reward > 0


def test_reward_leave_negative():
    cfg = RewardConfig()
    reward = calculate_reward(
        action_type="recommend",
        outcome="leave",
        repetition_level=1,
        done=True,
        config=cfg,
    )
    assert reward < 0


def test_reward_invalid_question_more_negative():
    cfg = RewardConfig()
    normal = calculate_reward(
        action_type="question",
        outcome="question",
        repetition_level=0,
        done=False,
        config=cfg,
    )
    invalid = calculate_reward(
        action_type="question",
        outcome="invalid_question",
        repetition_level=0,
        done=False,
        config=cfg,
    )
    assert invalid < normal


def test_repetition_penalty_increases_negative_reward():
    cfg = RewardConfig()
    low_rep = calculate_reward(
        action_type="recommend",
        outcome="skip",
        repetition_level=1,
        done=False,
        config=cfg,
    )
    high_rep = calculate_reward(
        action_type="recommend",
        outcome="skip",
        repetition_level=3,
        done=False,
        config=cfg,
    )
    assert high_rep < low_rep
