from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.agent.offline_rl import OfflineReplayEnv, OfflineTransition, load_offline_transitions
from src.config import load_config
from src.env.grid_env import GridEnv
from src.logging.db import DBLogger


def _obs_payload(action_count: int = 14) -> dict:
    return {
        "local_tiles": np.zeros((5, 5), dtype=np.int32).tolist(),
        "local_entities": np.zeros((4, 3), dtype=np.float32).tolist(),
        "hand_item": np.zeros((4,), dtype=np.float32).tolist(),
        "stats": [10.0, 1.0, 0.0, 1.0],
        "mode_features": [1.0, 0.0],
        "action_mask": [1] * action_count,
        "memory_hint": [0.0],
    }


def test_load_offline_transitions_from_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "atlas.db"
    logger = DBLogger(db_path)
    logger.start_episode("dungeon_exit", 7, "ExitGame", "2026-01-01T00:00:00")
    obs = _obs_payload()
    logger.log_step(obs, action=2, reward=1.25, done=False, info={"a": 1}, reward_terms={"mode": 1.0})

    transitions = load_offline_transitions(db_path)
    assert len(transitions) == 1
    tr = transitions[0]
    assert tr.mode == "ExitGame"
    assert tr.action == 2
    assert tr.reward == 1.25


def test_load_offline_transitions_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    row = {"mode": "HideAndSeek", "obs": _obs_payload(), "action": 3, "reward": 0.5, "done": True}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    transitions = load_offline_transitions(path)
    assert len(transitions) == 1
    assert transitions[0].mode == "HideAndSeek"
    assert transitions[0].done is True


def test_offline_replay_env_runs_step() -> None:
    config = load_config(None)
    env = GridEnv(config)
    transitions = [
        OfflineTransition(
            mode="ExitGame",
            obs=_obs_payload(action_count=env.action_space.n),
            action=2,
            reward=1.0,
            done=False,
        )
    ]
    replay_env = OfflineReplayEnv(
        transitions=transitions,
        observation_space=env.observation_space,
        action_space=env.action_space,
        algorithm="iql",
    )
    obs, _ = replay_env.reset(seed=3)
    assert "action_mask" in obs
    _obs2, reward, done, _trunc, info = replay_env.step(2)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info["algorithm"] == "iql"
