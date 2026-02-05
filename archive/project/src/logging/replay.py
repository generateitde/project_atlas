from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import create_engine, select

from src.logging.schema import steps


def export_replay(db_path: Path, output_path: Path) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        rows = conn.execute(select(steps)).fetchall()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row._mapping)) + "\n")
