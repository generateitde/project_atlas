from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.logging.schema import Step


def export_steps(db_path: Path, out_path: Path) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        steps = session.execute(select(Step)).scalars().all()
    with out_path.open("w", encoding="utf-8") as handle:
        for step in steps:
            handle.write(
                json.dumps(
                    {
                        "episode_id": step.episode_id,
                        "tick": step.tick,
                        "obs": step.obs_json,
                        "action": step.action_int,
                        "reward": step.reward_float,
                        "done": step.done_bool,
                        "info": step.info_json,
                    }
                )
                + "\n"
            )
