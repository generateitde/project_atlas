from __future__ import annotations

import hashlib
import json
from typing import Any

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO

from src.agent.action_masking import apply_action_mask
from src.config import AtlasConfig


def build_model(env, config: AtlasConfig) -> RecurrentPPO:
    policy_kwargs: dict[str, Any] = {
        "lstm_hidden_size": config.training.policy_hidden_size,
        "net_arch": [config.training.policy_hidden_size],
    }
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        ent_coef=config.training.ent_coef,
        learning_rate=config.training.learning_rate,
        clip_range=config.training.clip_range,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=config.training.seed,
    )
    return model


def observation_schema_signature(observation_space: gym.Space) -> dict[str, Any]:
    """Build a JSON-serialisable schema signature for observation compatibility checks."""

    if isinstance(observation_space, gym.spaces.Dict):
        return {
            "type": "dict",
            "keys": {
                key: observation_schema_signature(subspace)
                for key, subspace in sorted(observation_space.spaces.items(), key=lambda item: item[0])
            },
        }
    if isinstance(observation_space, gym.spaces.Box):
        return {
            "type": "box",
            "shape": list(observation_space.shape),
            "dtype": str(observation_space.dtype),
        }
    if isinstance(observation_space, gym.spaces.Discrete):
        return {
            "type": "discrete",
            "n": int(observation_space.n),
        }
    if isinstance(observation_space, gym.spaces.MultiBinary):
        return {
            "type": "multibinary",
            "n": int(observation_space.n),
        }
    if isinstance(observation_space, gym.spaces.MultiDiscrete):
        return {
            "type": "multidiscrete",
            "nvec": observation_space.nvec.tolist(),
        }
    return {"type": observation_space.__class__.__name__}


def schema_hash(schema: dict[str, Any]) -> str:
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def choose_masked_action(logits: np.ndarray, mask: np.ndarray) -> int:
    masked_logits = apply_action_mask(np.asarray(logits), np.asarray(mask, dtype=bool))
    return int(np.argmax(masked_logits))
