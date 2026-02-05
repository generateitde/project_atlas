from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from sb3_contrib import RecurrentPPO

from src.agent.trainer import AtlasTrainer
from src.config import AtlasConfig
from src.env.grid_env import GridEnv
from src.env.modes import mode_success


@dataclass(frozen=True)
class EvalScenario:
    mode: str
    mode_params: dict[str, Any]
    seed: int


@dataclass
class EpisodeMetrics:
    checkpoint: str
    mode: str
    seed: int
    success: bool
    episode_return: float
    steps: int
    steps_to_goal: int | None
    invalid_actions: int
    total_actions: int


@dataclass
class ModeAggregate:
    checkpoint: str
    mode: str
    episodes: int
    success_rate: float
    avg_return: float
    avg_steps_to_goal: float | None
    invalid_action_rate: float


class DeterministicEvalHarness:
    def __init__(self, config: AtlasConfig, checkpoint_dir: Path) -> None:
        self.config = config
        self.checkpoint_dir = checkpoint_dir

    def evaluate(
        self,
        *,
        checkpoints: list[Path],
        seeds: list[int],
        mode_matrix: list[tuple[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        scenarios = [EvalScenario(mode=mode, mode_params=params, seed=seed) for mode, params in mode_matrix for seed in seeds]
        rows: list[ModeAggregate] = []

        for checkpoint in checkpoints:
            episode_metrics = self._evaluate_checkpoint(checkpoint, scenarios)
            rows.extend(self._aggregate(episode_metrics))

        payload = {
            "checkpoints": [str(p) for p in checkpoints],
            "seeds": seeds,
            "mode_matrix": [{"mode": m, "params": p} for m, p in mode_matrix],
            "rows": [
                {
                    "checkpoint": row.checkpoint,
                    "mode": row.mode,
                    "episodes": row.episodes,
                    "success_rate": row.success_rate,
                    "avg_return": row.avg_return,
                    "avg_steps_to_goal": row.avg_steps_to_goal,
                    "invalid_action_rate": row.invalid_action_rate,
                }
                for row in rows
            ],
        }
        return payload

    def _evaluate_checkpoint(self, checkpoint: Path, scenarios: list[EvalScenario]) -> list[EpisodeMetrics]:
        env = GridEnv(self.config)
        trainer = AtlasTrainer(self.config, self.checkpoint_dir)
        trainer.load(env)
        if checkpoint.exists():
            trainer.model = RecurrentPPO.load(checkpoint, env=env)

        if trainer.model is None:
            raise RuntimeError("Model not initialized for evaluation")

        metrics: list[EpisodeMetrics] = []
        for scenario in scenarios:
            obs, _ = env.reset(seed=scenario.seed)
            env.set_mode(scenario.mode, scenario.mode_params)
            recurrent_state = None
            episode_start = True
            done = False
            episode_return = 0.0
            step_count = 0
            invalid_actions = 0
            total_actions = 0
            steps_to_goal: int | None = None

            while not done:
                action, recurrent_state = trainer.model.predict(
                    obs,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                action_int = int(action)
                action_mask = obs.get("action_mask")
                if action_mask is not None and action_int < len(action_mask) and not bool(action_mask[action_int]):
                    invalid_actions += 1
                total_actions += 1

                obs, reward, done, _, info = env.step(action_int)
                episode_return += float(reward)
                step_count += 1
                episode_start = bool(done)

            success = mode_success(env.mode.name, info)
            if success:
                steps_to_goal = step_count
            metrics.append(
                EpisodeMetrics(
                    checkpoint=str(checkpoint),
                    mode=scenario.mode,
                    seed=scenario.seed,
                    success=success,
                    episode_return=episode_return,
                    steps=step_count,
                    steps_to_goal=steps_to_goal,
                    invalid_actions=invalid_actions,
                    total_actions=total_actions,
                )
            )
        return metrics

    def _aggregate(self, episodes: list[EpisodeMetrics]) -> list[ModeAggregate]:
        grouped: dict[tuple[str, str], list[EpisodeMetrics]] = {}
        for item in episodes:
            grouped.setdefault((item.checkpoint, item.mode), []).append(item)

        rows: list[ModeAggregate] = []
        for (checkpoint, mode), items in grouped.items():
            successes = [item.success for item in items]
            success_steps = [item.steps_to_goal for item in items if item.steps_to_goal is not None]
            invalid_total = sum(item.invalid_actions for item in items)
            action_total = sum(item.total_actions for item in items)
            rows.append(
                ModeAggregate(
                    checkpoint=checkpoint,
                    mode=mode,
                    episodes=len(items),
                    success_rate=float(sum(successes) / max(1, len(items))),
                    avg_return=float(mean(item.episode_return for item in items)),
                    avg_steps_to_goal=float(mean(success_steps)) if success_steps else None,
                    invalid_action_rate=float(invalid_total / max(1, action_total)),
                )
            )
        rows.sort(key=lambda row: (row.checkpoint, row.mode))
        return rows

