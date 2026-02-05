from src.env.rewards import compute_reward


def test_compute_reward_includes_preference_term() -> None:
    reward, breakdown = compute_reward(mode_reward=0.0, events=[], preference_reward=0.4)
    assert breakdown["preference"] == 0.4
    assert reward > 0.0
