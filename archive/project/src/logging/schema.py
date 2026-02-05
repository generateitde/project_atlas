from __future__ import annotations

from sqlalchemy import Boolean, Column, Float, Integer, MetaData, String, Table

metadata = MetaData()

episodes = Table(
    "episodes",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("preset", String),
    Column("seed", Integer),
    Column("mode", String),
    Column("started_at", String),
)

steps = Table(
    "steps",
    metadata,
    Column("episode_id", Integer),
    Column("tick", Integer),
    Column("obs_json", String),
    Column("action_int", Integer),
    Column("action_json", String),
    Column("reward_float", Float),
    Column("done_bool", Boolean),
    Column("info_json", String),
)

events = Table(
    "events",
    metadata,
    Column("episode_id", Integer),
    Column("tick", Integer),
    Column("type", String),
    Column("payload_json", String),
)

human_actions = Table(
    "human_actions",
    metadata,
    Column("episode_id", Integer),
    Column("tick", Integer),
    Column("action_int", Integer),
    Column("obs_json", String),
)

atlas_messages = Table(
    "atlas_messages",
    metadata,
    Column("episode_id", Integer),
    Column("tick", Integer),
    Column("msg_type", String),
    Column("text", String),
    Column("metadata_json", String),
)

human_feedback = Table(
    "human_feedback",
    metadata,
    Column("episode_id", Integer),
    Column("tick", Integer),
    Column("target", String),
    Column("msg_type", String),
    Column("score", Integer),
    Column("correction_text", String),
)
