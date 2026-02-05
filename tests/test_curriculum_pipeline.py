from src.agent.trainer import CurriculumManager
from src.env.modes import CurriculumStage, default_curriculum_stages


def test_curriculum_defaults_have_expected_progression() -> None:
    stages = default_curriculum_stages()
    assert [stage.name for stage in stages] == ["exit_basic", "exit_hazards", "ctf_basic"]
    assert stages[0].mode == "ExitGame"
    assert stages[-1].mode == "CaptureTheFlag"


def test_curriculum_stage_transitions_when_thresholds_met() -> None:
    manager = CurriculumManager(
        stages=[
            CurriculumStage(
                name="stage_a",
                preset="dungeon_exit",
                mode="ExitGame",
                mode_params={},
                min_episodes=2,
                min_success_rate=0.5,
                min_avg_return=2.0,
            ),
            CurriculumStage(
                name="stage_b",
                preset="ctf_small",
                mode="CaptureTheFlag",
                mode_params={},
                min_episodes=2,
                min_success_rate=0.5,
                min_avg_return=1.0,
            ),
        ]
    )

    transitioned_1, reason_1 = manager.record_episode(success=True, episode_return=2.5)
    assert transitioned_1 is False
    assert reason_1 is None
    transitioned_2, reason_2 = manager.record_episode(success=True, episode_return=2.5)
    assert transitioned_2 is True
    assert reason_2 is not None
    assert "success_rate=" in reason_2
    assert manager.current_stage().name == "stage_b"
