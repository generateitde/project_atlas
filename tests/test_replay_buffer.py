from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.agent.replay_buffer import MultiModeReplayBuffer
from src.logging.db import DBLogger
from src.logging.schema import ReplayBufferStat


def _obs(step: int) -> dict:
    return {"step": step, "action_mask": [1] * 12}


def test_mode_balanced_sampling_is_near_even_over_many_samples() -> None:
    buffer = MultiModeReplayBuffer(capacity=2000, rng_seed=7)
    for i in range(600):
        buffer.add(mode="ExitGame", obs=_obs(i), action=1, reward=0.1, next_obs=_obs(i + 1), done=False)
    for i in range(300):
        buffer.add(mode="CaptureTheFlag", obs=_obs(i), action=2, reward=0.2, next_obs=_obs(i + 1), done=False)
    for i in range(100):
        buffer.add(mode="HideAndSeek", obs=_obs(i), action=3, reward=0.3, next_obs=_obs(i + 1), done=False)

    counts = {"ExitGame": 0, "CaptureTheFlag": 0, "HideAndSeek": 0}
    for _ in range(1000):
        batch = buffer.sample(batch_size=10, strategy="mode-balanced")
        for sample in batch:
            counts[sample.mode] += 1

    total = sum(counts.values())
    expected = total / 3
    for mode_count in counts.values():
        assert abs(mode_count - expected) / expected < 0.10


def test_replay_buffer_stats_are_logged_to_db(tmp_path: Path) -> None:
    db_path = tmp_path / "atlas.db"
    logger = DBLogger(db_path)
    logger.start_episode("dungeon_exit", 11, "ExitGame", "2026-01-01T00:00:00")

    stats = {
        "total_transitions": 42,
        "sample_entropy": 1.08,
        "mode_coverage": {"ExitGame": 0.5, "CaptureTheFlag": 0.5},
    }
    logger.log_replay_buffer_stats(stats)

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        row = session.query(ReplayBufferStat).first()
        assert row is not None
        assert row.total_transitions == 42
        assert row.sample_entropy > 0.0
