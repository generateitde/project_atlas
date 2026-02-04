from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class ToggleConfig(BaseModel):
    use_llm: bool
    use_voice_in: bool
    use_voice_out: bool
    push_to_talk: bool
    speak_agent_messages: bool


class RenderingConfig(BaseModel):
    tile_size: int
    fps: int


class TimingConfig(BaseModel):
    env_step_hz: int
    agent_step_hz: int


class TrainingConfig(BaseModel):
    algo: Literal["recurrent_ppo"]
    n_envs: int
    seed: int
    save_every_steps: int
    eval_every_steps: int
    gamma: float
    gae_lambda: float
    ent_coef: float
    learning_rate: float
    clip_range: float
    n_steps: int
    batch_size: int
    policy_hidden_size: int


class WorldConfig(BaseModel):
    width: int
    height: int
    visibility_radius: int
    max_episode_steps: int


class ProgressionConfig(BaseModel):
    enable_leveling: bool
    exp_curve: Literal["linear", "sqrt", "exp"]
    exp_per_kill: int
    gate_enabled: bool


class OpenAIConfig(BaseModel):
    api_key_env: str
    llm_model: str
    stt_model: str
    tts_model: str


class AtlasConfig(BaseModel):
    toggles: ToggleConfig
    rendering: RenderingConfig
    timing: TimingConfig
    training: TrainingConfig
    world: WorldConfig
    progression: ProgressionConfig
    openai: OpenAIConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> AtlasConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return AtlasConfig.model_validate(data)
