from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GoalStackState:
    mode_goal: str
    subgoals: list[str]

    @property
    def active_subgoal(self) -> str:
        if not self.subgoals:
            return "explore"
        return self.subgoals[0]


@dataclass
class WorldModel:
    visited: set[tuple[int, int]] = field(default_factory=set)
    frontier: set[tuple[int, int]] = field(default_factory=set)
    breakable_tested: dict[tuple[int, int], bool] = field(default_factory=dict)
    boundaries: set[tuple[int, int]] = field(default_factory=set)


@dataclass
class GoalManager:
    state: GoalStackState = field(default_factory=lambda: GoalStackState(mode_goal="Explore", subgoals=["explore"]))

    def update(self, mode_name: str, mode_info: dict[str, Any] | None = None) -> GoalStackState:
        mode_info = mode_info or {}
        objective = str(mode_info.get("objective") or mode_name)
        if mode_name == "ExitGame":
            self.state = self._exit_game_goals(objective, mode_info)
        elif mode_name == "CaptureTheFlag":
            self.state = self._ctf_goals(objective, mode_info)
        elif mode_name == "HideAndSeek":
            self.state = GoalStackState(mode_goal=objective, subgoals=["find_hide_target"])
        else:
            self.state = GoalStackState(mode_goal=objective, subgoals=["explore"])
        return self.state

    def _exit_game_goals(self, objective: str, mode_info: dict[str, Any]) -> GoalStackState:
        if mode_info.get("done"):
            return GoalStackState(mode_goal=objective, subgoals=["goal_reached"])
        if not bool(mode_info.get("key_collected")):
            return GoalStackState(mode_goal=objective, subgoals=["find_key", "go_door", "go_goal"])
        if not bool(mode_info.get("door_open")):
            return GoalStackState(mode_goal=objective, subgoals=["go_door", "go_goal"])
        return GoalStackState(mode_goal=objective, subgoals=["go_goal"])

    def _ctf_goals(self, objective: str, mode_info: dict[str, Any]) -> GoalStackState:
        if mode_info.get("done"):
            return GoalStackState(mode_goal=objective, subgoals=["score_complete"])
        if bool(mode_info.get("atlas_has_flag")):
            return GoalStackState(mode_goal=objective, subgoals=["return_flag", "score"])
        return GoalStackState(mode_goal=objective, subgoals=["find_flag", "return_flag", "score"])
