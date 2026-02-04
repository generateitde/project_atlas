from __future__ import annotations

from typing import Any

from sb3_contrib import RecurrentPPO

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
