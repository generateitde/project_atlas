from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict

from src.env.grid_env import GridEnv
from src.env.world_gen import PRESETS


@dataclass
class ConsoleState:
    open: bool = False
    buffer: str = ""


class Console:
    def __init__(self, env: GridEnv) -> None:
        self.env = env
        self.state = ConsoleState()
        self.commands: Dict[str, Callable[[list[str]], str]] = {
            "help": self._help,
            "world": self._world,
            "mode": self._mode,
            "goal": self._goal,
            "teleport": self._teleport,
            "pause": self._pause,
            "resume": self._resume,
            "reset": self._reset,
            "print": self._print,
        }
        self.ai_paused = False

    def execute(self, line: str) -> str:
        parts = line.strip().split()
        if not parts:
            return ""
        cmd = parts[0]
        handler = self.commands.get(cmd)
        if not handler:
            return "Unknown command"
        return handler(parts[1:])

    def _help(self, args: list[str]) -> str:
        return "Commands: help, world list|switch <preset> <seed>, mode set <mode>, goal set <json>, teleport <human|ai_atlas> <x> <y>, pause ai, resume ai, reset episode, print state|map"

    def _world(self, args: list[str]) -> str:
        if not args:
            return "world list|switch <preset> <seed>"
        if args[0] == "list":
            return ", ".join(PRESETS)
        if args[0] == "switch" and len(args) >= 3:
            preset = args[1]
            seed = int(args[2])
            self.env.switch_world(preset, seed)
            return f"Switched to {preset} seed {seed}"
        return "Invalid world command"

    def _mode(self, args: list[str]) -> str:
        if len(args) >= 2 and args[0] == "set":
            self.env.set_mode(args[1])
            return f"Mode set to {args[1]}"
        return "mode set <mode>"

    def _goal(self, args: list[str]) -> str:
        if len(args) >= 2 and args[0] == "set":
            return f"Goal set: {json.dumps(args[1:])}"
        return "goal set <json_goal>"

    def _teleport(self, args: list[str]) -> str:
        if len(args) >= 3:
            target = args[0]
            x = int(args[1])
            y = int(args[2])
            if target == "human":
                self.env.world.human.pos = (x, y)
            else:
                self.env.world.atlas.pos = (x, y)
            return f"Teleported {target} to {x},{y}"
        return "teleport <human|ai_atlas> <x> <y>"

    def _pause(self, args: list[str]) -> str:
        if args and args[0] == "ai":
            self.ai_paused = True
            return "AI paused"
        return "pause ai"

    def _resume(self, args: list[str]) -> str:
        if args and args[0] == "ai":
            self.ai_paused = False
            return "AI resumed"
        return "resume ai"

    def _reset(self, args: list[str]) -> str:
        if args and args[0] == "episode":
            self.env.reset()
            return "Episode reset"
        return "reset episode"

    def _print(self, args: list[str]) -> str:
        if not args:
            return "print state|map"
        if args[0] == "state":
            return f"Human: {self.env.world.human.pos}, Atlas: {self.env.world.atlas.pos}"
        if args[0] == "map":
            return "\n".join("".join(tile.value[0] for tile in row) for row in self.env.world.tiles)
        return "print state|map"
