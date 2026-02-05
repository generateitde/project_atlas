from __future__ import annotations

from sqlalchemy import Boolean, Column, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Episode(Base):
    __tablename__ = "episodes"
    id = Column(Integer, primary_key=True)
    preset = Column(String(64))
    seed = Column(Integer)
    mode = Column(String(64))
    started_at = Column(String(64))


class Step(Base):
    __tablename__ = "steps"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer)
    tick = Column(Integer)
    obs_json = Column(Text)
    action_int = Column(Integer)
    action_json = Column(Text)
    reward_float = Column(Float)
    reward_terms_json = Column(Text)
    done_bool = Column(Boolean)
    info_json = Column(Text)


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer)
    tick = Column(Integer)
    type = Column(String(64))
    payload_json = Column(Text)


class HumanAction(Base):
    __tablename__ = "human_actions"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer)
    tick = Column(Integer)
    action_int = Column(Integer)
    obs_json = Column(Text)


class AtlasMessage(Base):
    __tablename__ = "atlas_messages"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer)
    tick = Column(Integer)
    msg_type = Column(String(64))
    text = Column(Text)
    metadata_json = Column(Text)


class HumanFeedback(Base):
    __tablename__ = "human_feedback"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer)
    tick = Column(Integer)
    target = Column(String(64))
    msg_type = Column(String(64))
    score = Column(Integer)
    correction_text = Column(Text)
