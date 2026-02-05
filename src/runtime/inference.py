from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sb3_contrib import RecurrentPPO

from src.agent.policy import observation_schema_signature, schema_hash

ARTIFACT_VERSION = "1"
MANIFEST_FILE = "manifest.json"
MODEL_FILE = "policy.zip"


@dataclass
class RuntimePolicy:
    model: RecurrentPPO
    recurrent_state: Any = None
    episode_start: bool = True

    def predict(self, obs: dict[str, Any], deterministic: bool = True):
        action, next_state = self.model.predict(
            obs,
            state=self.recurrent_state,
            episode_start=self.episode_start,
            deterministic=deterministic,
        )
        self.recurrent_state = next_state
        self.episode_start = False
        return int(action)

    def mark_episode_done(self) -> None:
        self.episode_start = True



def export_policy_artifact(model_path: Path, env, out_dir: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    exported_model_path = out_dir / MODEL_FILE
    shutil.copy2(model_path, exported_model_path)

    obs_schema = observation_schema_signature(env.observation_space)
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_file": MODEL_FILE,
        "observation_schema": obs_schema,
        "observation_schema_hash": schema_hash(obs_schema),
    }

    manifest_path = out_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path



def load_runtime_policy(artifact_dir: Path, env) -> RuntimePolicy:
    manifest_path = artifact_dir / MANIFEST_FILE
    model_path = artifact_dir / MODEL_FILE

    if not manifest_path.exists():
        raise FileNotFoundError(f"Policy artifact missing manifest: {manifest_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Policy artifact missing model: {model_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_schema = manifest.get("observation_schema")
    expected_hash = manifest.get("observation_schema_hash")

    runtime_schema = observation_schema_signature(env.observation_space)
    runtime_hash = schema_hash(runtime_schema)

    if expected_schema != runtime_schema or expected_hash != runtime_hash:
        raise ValueError(
            "Observation schema mismatch for runtime inference: "
            f"artifact_hash={expected_hash}, runtime_hash={runtime_hash}."
        )

    model = RecurrentPPO.load(model_path, env=env)
    return RuntimePolicy(model=model)
