from src.config import load_config
from src.env import encoding
from src.env.grid_env import GridEnv


def test_forbidden_chain_rejects_ask_human_after_speak_with_reason():
    env = GridEnv(load_config(), strict_safety=True)
    env.reset(seed=123)

    _, _, _, _, info_speak = env.step(12)
    assert info_speak["tool_safety"]["rejected"] is False

    _, reward, _, _, info = env.step(13)
    assert info["tool_safety"]["rejected"] is True
    assert info["tool_safety"]["rejection_code"] == "forbidden_chain"
    assert info["reward_terms"]["safety_penalty"] == -0.05
    assert isinstance(reward, float)


def test_strict_safety_masks_tool_spam():
    env = GridEnv(load_config(), strict_safety=True)
    obs, _ = env.reset(seed=1)

    for _ in range(3):
        obs, _, _, _, _ = env.step(11)

    assert obs["action_mask"][11] == 0
    assert encoding.ACTION_MEANINGS[11] == "INSPECT_ADJACENT"
