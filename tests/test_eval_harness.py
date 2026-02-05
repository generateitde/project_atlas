from __future__ import annotations

import json
from pathlib import Path

from src.eval.harness import DeterministicEvalHarness
from src.logging.db import write_eval_trend_report


def test_eval_harness_is_deterministic_given_same_checkpoint(monkeypatch) -> None:
    harness = DeterministicEvalHarness(config=None, checkpoint_dir=Path("checkpoints"))  # type: ignore[arg-type]

    canned = [
        {
            "checkpoint": "checkpoints/atlas_model.zip",
            "mode": "ExitGame",
            "episodes": 3,
            "success_rate": 2 / 3,
            "avg_return": 1.5,
            "avg_steps_to_goal": 42.0,
            "invalid_action_rate": 0.0,
        }
    ]

    def fake_eval_checkpoint(_checkpoint, _scenarios):
        return []

    def fake_aggregate(_episodes):
        from src.eval.harness import ModeAggregate

        return [
            ModeAggregate(
                checkpoint=canned[0]["checkpoint"],
                mode=canned[0]["mode"],
                episodes=canned[0]["episodes"],
                success_rate=canned[0]["success_rate"],
                avg_return=canned[0]["avg_return"],
                avg_steps_to_goal=canned[0]["avg_steps_to_goal"],
                invalid_action_rate=canned[0]["invalid_action_rate"],
            )
        ]

    monkeypatch.setattr(harness, "_evaluate_checkpoint", fake_eval_checkpoint)
    monkeypatch.setattr(harness, "_aggregate", fake_aggregate)

    params = dict(checkpoints=[Path("checkpoints/atlas_model.zip")], seeds=[1, 2, 3], mode_matrix=[("ExitGame", {})])
    report_a = harness.evaluate(**params)
    report_b = harness.evaluate(**params)

    assert report_a == report_b
    assert report_a["rows"] == canned


def test_write_eval_trend_report_outputs_json_and_csv(tmp_path: Path) -> None:
    rows = [
        {
            "checkpoint": "checkpoints/atlas_model.zip",
            "mode": "ExitGame",
            "episodes": 5,
            "success_rate": 0.8,
            "avg_return": 2.5,
            "avg_steps_to_goal": 31.25,
            "invalid_action_rate": 0.001,
        }
    ]
    json_path = tmp_path / "eval_trends.json"
    csv_path = tmp_path / "eval_trends.csv"

    write_eval_trend_report(rows, json_path, csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["mode"] == "ExitGame"

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "checkpoint,mode,episodes,success_rate,avg_return,avg_steps_to_goal,invalid_action_rate" in csv_text
    assert "ExitGame" in csv_text
