from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.logging.schema import Base, Episode, Event, HumanAction, HumanFeedback, ReplayBufferStat, Step


class DBLogger:
    def __init__(self, path: Path) -> None:
        self.engine = create_engine(f"sqlite:///{path}")
        Base.metadata.create_all(self.engine)
        self.episode_id: int | None = None
        self.tick = 0

    def start_episode(
        self,
        preset: str,
        seed: int,
        mode: str,
        started_at: str,
        world_hash: str | None = None,
        curriculum_stage: str | None = None,
        stage_transition_reason: str | None = None,
    ) -> None:
        with Session(self.engine) as session:
            episode = Episode(
                preset=preset,
                seed=seed,
                mode=mode,
                started_at=started_at,
                world_hash=world_hash,
                curriculum_stage=curriculum_stage,
                stage_transition_reason=stage_transition_reason,
            )
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

    def log_replay_buffer_stats(self, stats: dict[str, Any]) -> None:
        if self.episode_id is None:
            return
        with Session(self.engine) as session:
            row = ReplayBufferStat(
                episode_id=self.episode_id,
                tick=self.tick,
                total_transitions=int(stats.get("total_transitions", 0)),
                sample_entropy=float(stats.get("sample_entropy", 0.0)),
                mode_coverage_json=json.dumps(stats.get("mode_coverage", {}), default=str),
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


def write_eval_trend_report(rows: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = "checkpoint,mode,episodes,success_rate,avg_return,avg_steps_to_goal,invalid_action_rate\n"
    lines = [header]
    for row in rows:
        lines.append(
            "{checkpoint},{mode},{episodes},{success_rate:.6f},{avg_return:.6f},{avg_steps_to_goal},{invalid_action_rate:.6f}\n".format(
                checkpoint=row["checkpoint"],
                mode=row["mode"],
                episodes=row["episodes"],
                success_rate=row["success_rate"],
                avg_return=row["avg_return"],
                avg_steps_to_goal="" if row["avg_steps_to_goal"] is None else f"{row['avg_steps_to_goal']:.6f}",
                invalid_action_rate=row["invalid_action_rate"],
            )
        )
    csv_path.write_text("".join(lines), encoding="utf-8")
