import numpy as np

from src.agent.preference_reward import PreferenceRewardModel, extract_state_features, parse_scored_feedback


def test_parse_scored_feedback() -> None:
    score, text = parse_scored_feedback("score:+2 great plan")
    assert score == 2
    assert text == "great plan"


def test_extract_state_features_shape() -> None:
    obs = {
        "local_tiles": np.ones((3, 3), dtype=np.int32),
        "stats": np.array([90.0, 2.0, 20.0, 3.0], dtype=np.float32),
        "action_mask": np.array([1, 1, 0, 1], dtype=np.int8),
    }
    features = extract_state_features(obs)
    assert features.shape == (6,)
    assert float(features[-1]) > 0.0


def test_preference_model_learns_signal() -> None:
    model = PreferenceRewardModel(text_bins=16)
    state = np.array([0.2, 0.8, 0.1, 0.0, 0.2, 0.5], dtype=np.float32)
    for _ in range(6):
        model.add_feedback(state, "good move", 2)
        model.add_feedback(state, "bad move", -2)
    model.train(epochs=40, lr=0.1, min_samples=2)
    good = model.score(state, "good move")
    bad = model.score(state, "bad move")
    assert good > bad
