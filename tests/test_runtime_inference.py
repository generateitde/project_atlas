from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from src.runtime.inference import export_policy_artifact, load_runtime_policy


class DummyEnv:
    def __init__(self, obs_shape: tuple[int, ...] = (3,)) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "stats": gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32),
                "action_mask": gym.spaces.MultiBinary(4),
            }
        )


def test_export_policy_artifact_writes_manifest_and_model(tmp_path: Path) -> None:
    checkpoint = tmp_path / "atlas_model.zip"
    checkpoint.write_bytes(b"dummy-model-bytes")
    artifact_dir = tmp_path / "artifact"

    manifest_path = export_policy_artifact(checkpoint, DummyEnv(), artifact_dir)

    assert manifest_path.exists()
    assert (artifact_dir / "policy.zip").exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["observation_schema_hash"]
    assert payload["model_file"] == "policy.zip"


def test_load_runtime_policy_fails_on_schema_mismatch(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "atlas_model.zip"
    checkpoint.write_bytes(b"dummy-model-bytes")
    artifact_dir = tmp_path / "artifact"
    export_policy_artifact(checkpoint, DummyEnv(obs_shape=(3,)), artifact_dir)

    monkeypatch.setattr("src.runtime.inference.RecurrentPPO.load", lambda *_args, **_kwargs: object())

    with pytest.raises(ValueError, match="Observation schema mismatch"):
        load_runtime_policy(artifact_dir, DummyEnv(obs_shape=(4,)))


def test_load_runtime_policy_uses_exported_model(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "atlas_model.zip"
    checkpoint.write_bytes(b"dummy-model-bytes")
    artifact_dir = tmp_path / "artifact"
    export_policy_artifact(checkpoint, DummyEnv(), artifact_dir)

    class StubModel:
        def predict(self, _obs, state=None, episode_start=True, deterministic=True):
            return 2, state

    monkeypatch.setattr("src.runtime.inference.RecurrentPPO.load", lambda *_args, **_kwargs: StubModel())

    runtime = load_runtime_policy(artifact_dir, DummyEnv())
    action = runtime.predict({"stats": np.zeros(3, dtype=np.float32), "action_mask": np.array([1, 1, 1, 1])})
    assert action == 2
