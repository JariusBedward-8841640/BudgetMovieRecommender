import numpy as np

from env.user_simulator import GENRES, PROFILE_NAMES, SimulatedUser, sample_profile_name


def test_profiles_constructible():
    rng = np.random.default_rng(1)
    for name in PROFILE_NAMES:
        user = SimulatedUser(name, rng)
        assert user.profile_name == name
        assert user.preference_scores.shape[0] == len(GENRES)


def test_question_answers_are_binary():
    rng = np.random.default_rng(2)
    user = SimulatedUser("balanced_viewer", rng)
    for question_id in (0, 1, 2):
        answer, _left = user.ask_question(question_id)
        assert answer in (-1, 1)


def test_recommend_outcome_valid():
    rng = np.random.default_rng(3)
    user = SimulatedUser("novelty_seeking", rng)
    outcome = user.recommend(genre_index=0, repetition_level=2)
    assert outcome in ("accept", "skip", "leave")


def test_sample_profile_name_respects_subset():
    rng = np.random.default_rng(4)
    subset = ["novelty_seeking", "question_sensitive"]
    sampled = sample_profile_name(rng, subset)
    assert sampled in subset
