from __future__ import annotations

from sb3_contrib import RecurrentPPO


def build_policy_kwargs(hidden_size: int) -> dict:
    return {
        "net_arch": [hidden_size, hidden_size],
    }


def build_model(env, config) -> RecurrentPPO:
    return RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        learning_rate=config.learning_rate,
        clip_range=config.clip_range,
        policy_kwargs=build_policy_kwargs(config.policy_hidden_size),
        verbose=0,
        seed=config.seed,
    )
