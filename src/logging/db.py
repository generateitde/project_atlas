from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.logging.schema import Base, Episode, Event, HumanAction, HumanFeedback, Step


class DBLogger:
    def __init__(self, path: Path) -> None:
        self.engine = create_engine(f"sqlite:///{path}")
        Base.metadata.create_all(self.engine)
        self.episode_id: int | None = None
        self.tick = 0

    def start_episode(self, preset: str, seed: int, mode: str, started_at: str, world_hash: str | None = None) -> None:
        with Session(self.engine) as session:
            episode = Episode(preset=preset, seed=seed, mode=mode, started_at=started_at, world_hash=world_hash)
            session.add(episode)
            session.commit()
            self.episode_id = episode.id
            self.tick = 0

    def log_step(
        self,
        obs: dict[str, Any],
        action: int,
        reward: float,
        done: bool,
        info: dict[str, Any],
        reward_terms: dict[str, Any] | None = None,
    ) -> None:
        if self.episode_id is None:
            return
        with Session(self.engine) as session:
            step = Step(
                episode_id=self.episode_id,
                tick=self.tick,
                obs_json=json.dumps(obs, default=str),
                action_int=action,
                action_json=json.dumps({"action": action}),
                reward_float=reward,
                reward_terms_json=json.dumps(reward_terms, default=str) if reward_terms is not None else None,
                done_bool=done,
                info_json=json.dumps(info, default=str),
            )
            session.add(step)
            session.commit()
        self.tick += 1

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.episode_id is None:
            return
        with Session(self.engine) as session:
            event = Event(
                episode_id=self.episode_id,
                tick=self.tick,
                type=event_type,
                payload_json=json.dumps(payload, default=str),
            )
            session.add(event)
            session.commit()


    def log_human_feedback(
        self,
        target: str,
        msg_type: str,
        score: int,
        correction_text: str,
        state_features: list[float] | None = None,
    ) -> None:
        if self.episode_id is None:
            return
        with Session(self.engine) as session:
            row = HumanFeedback(
                episode_id=self.episode_id,
                tick=self.tick,
                target=target,
                msg_type=msg_type,
                score=int(score),
                correction_text=correction_text,
                state_features_json=json.dumps(state_features or []),
            )
            session.add(row)
            session.commit()

    def log_human_action(self, obs: dict[str, Any], action: int) -> None:
        if self.episode_id is None:
            return
        with Session(self.engine) as session:
            row = HumanAction(
                episode_id=self.episode_id,
                tick=self.tick,
                action_int=int(action),
                obs_json=json.dumps(obs, default=str),
            )
            session.add(row)
            session.commit()
