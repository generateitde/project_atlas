import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.agent.imitation import ImitationBuffer
from src.logging.db import DBLogger
from src.logging.schema import HumanAction


def _obs(action_mask=None):
    return {
        "local_tiles": [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
        "mode_features": [1.0, 0.0],
        "stats": [10.0, 5.0, 0.0, 0.0],
        "action_mask": action_mask if action_mask is not None else [1] * 14,
    }


def test_imitation_buffer_prefers_human_majority_action():
    buffer = ImitationBuffer(min_confidence=0.7)
    for _ in range(4):
        buffer.add(_obs(), 2)
    buffer.add(_obs(), 4)

    assert buffer.select_action(_obs(), rl_action=4) == 2


def test_imitation_respects_action_mask():
    buffer = ImitationBuffer(min_confidence=0.6)
    for _ in range(3):
        buffer.add(_obs(), 10)

    mask = [1] * 14
    mask[10] = 0
    assert buffer.select_action(_obs(action_mask=mask), rl_action=2) == 2


def test_db_logger_persists_human_actions(tmp_path: Path):
    db_path = tmp_path / "atlas_test.db"
    logger = DBLogger(db_path)
    logger.start_episode("preset", 123, "ExitGame", "2024-01-01T00:00:00", "hash")
    obs = _obs()
    logger.log_human_action(obs, 5)

    with Session(logger.engine) as session:
        row = session.execute(select(HumanAction)).scalar_one()

    assert row.action_int == 5
    payload = json.loads(row.obs_json)
    assert payload["local_tiles"][1][1] == 2
