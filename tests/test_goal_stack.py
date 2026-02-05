from src.agent.world_model import GoalManager


def test_exitgame_subgoal_sequence() -> None:
    manager = GoalManager()

    phase_find_key = manager.update(
        "ExitGame",
        {
            "objective": "Collect key, pass door, reach exit tile.",
            "key_collected": False,
            "door_open": False,
            "done": False,
        },
    )
    assert phase_find_key.active_subgoal == "find_key"

    phase_door = manager.update(
        "ExitGame",
        {
            "objective": "Collect key, pass door, reach exit tile.",
            "key_collected": True,
            "door_open": False,
            "done": False,
        },
    )
    assert phase_door.active_subgoal == "go_door"

    phase_goal = manager.update(
        "ExitGame",
        {
            "objective": "Collect key, pass door, reach exit tile.",
            "key_collected": True,
            "door_open": True,
            "done": False,
        },
    )
    assert phase_goal.active_subgoal == "go_goal"


def test_ctf_subgoals_switch_with_flag_state() -> None:
    manager = GoalManager()

    no_flag = manager.update(
        "CaptureTheFlag",
        {"objective": "Capture the flag and return it to score.", "atlas_has_flag": False, "done": False},
    )
    assert no_flag.active_subgoal == "find_flag"

    has_flag = manager.update(
        "CaptureTheFlag",
        {"objective": "Capture the flag and return it to score.", "atlas_has_flag": True, "done": False},
    )
    assert has_flag.active_subgoal == "return_flag"
