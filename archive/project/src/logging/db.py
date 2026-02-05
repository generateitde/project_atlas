from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.logging.schema import metadata, steps


class Database:
    def __init__(self, path: Path) -> None:
        self.engine: Engine = create_engine(f"sqlite:///{path}")
        metadata.create_all(self.engine)

    def log_step(
        self,
        episode_id: int,
        tick: int,
        obs: Dict[str, Any],
        action_int: int,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                steps.insert().values(
                    episode_id=episode_id,
                    tick=tick,
                    obs_json=json.dumps(obs, default=str),
                    action_int=action_int,
                    action_json=json.dumps({"action": action_int}),
                    reward_float=reward,
                    done_bool=done,
                    info_json=json.dumps(info, default=str),
                )
            )
