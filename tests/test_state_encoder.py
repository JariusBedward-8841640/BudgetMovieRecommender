import numpy as np

from env.state_encoder import EncoderConfig, encode_tabular, encode_vector, tabular_state_space_size


def test_tabular_state_id_within_space():
    cfg = EncoderConfig(max_steps=8, question_budget=2, num_genres=5)
    space_size = tabular_state_space_size(cfg)
    state_id = encode_tabular(
        step_index=3,
        max_steps=8,
        remaining_budget=1,
        question_budget=2,
        engagement=0.5,
        belief_scores=np.array([0.1, 0.2, 0.0, -0.1, 0.3], dtype=np.float32),
        uncertainty=0.6,
        last_action_type="recommend",
        recent_outcome="skip",
        repetition_level=2,
    )
    assert 0 <= state_id < space_size


def test_vector_encoding_shape_and_dtype():
    vector = encode_vector(
        step_index=2,
        max_steps=8,
        remaining_budget=1,
        question_budget=2,
        engagement=0.7,
        belief_scores=np.array([0.0, 0.1, 0.2, -0.2, 0.3], dtype=np.float32),
        uncertainty=0.4,
        last_action_type="question",
        recent_outcome="accept",
        repetition_level=1,
    )
    assert vector.shape == (17,)
    assert vector.dtype == np.float32
