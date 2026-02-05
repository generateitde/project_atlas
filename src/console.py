from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.env.modes import create_mode
from src.env.world_gen import PRESETS


@dataclass
class Console:
    active: bool = False
    buffer: str = ""
    last_message: str = ""
    help_text: str = field(
        default_factory=lambda: (
            "help | seed set <seed> | world list | world switch <preset> <seed> | mode set <mode> <json_params> | "
            "goal set <text> | ai mode <explore|query> | control <human|ai_atlas> | enemy spawn <type> <x> <y> <hp> <exp> | "
            "item spawn <type> <x> <y> | teleport <human|ai_atlas> <x> <y> | pause ai | resume ai | "
            "save | load | reset episode | print state | print map | agent info"
        )
    )

    def execute(self, game, command: str) -> str:
        parts = command.strip().split()
        if not parts:
            return ""
        if parts[0] == "help":
            return self.help_text
        if parts[:2] == ["seed", "set"] and len(parts) >= 3:
            seed = int(parts[2])
            game.set_seed(seed)
            return f"seed set to {seed}"
        if parts[:2] == ["world", "list"]:
            return ", ".join(PRESETS.keys())
        if parts[:2] == ["world", "switch"] and len(parts) >= 4:
            preset = parts[2]
            seed = int(parts[3])
            game.switch_world(preset, seed)
            return f"switched world to {preset} seed {seed}"
        if parts[:2] == ["mode", "set"] and len(parts) >= 3:
            mode_name = parts[2]
            params = json.loads(" ".join(parts[3:]) or "{}")
            game.env.set_mode(mode_name, params)
            return f"mode set to {mode_name}"
        if parts[:2] == ["goal", "set"] and len(parts) >= 3:
            raw_goal = " ".join(parts[2:]).strip()
            goal_text = raw_goal
            if raw_goal.startswith("{"):
                try:
                    payload = json.loads(raw_goal)
                    goal_text = payload.get("text", raw_goal)
                except json.JSONDecodeError:
                    goal_text = raw_goal
            game.goal_text = goal_text
            return f"goal set to: {goal_text}"
        if parts[:2] == ["ai", "mode"] and len(parts) >= 3:
            mode = parts[2].lower()
            if mode in {"explore", "query"}:
                game.ai_mode = mode
                game.waiting_for_response = False
                game.steps_since_question = 0
                return f"AI mode set to {mode}"
            return "unknown AI mode (use explore or query)"
        if parts[0] == "control" and len(parts) >= 2:
            target = parts[1]
            if target in {"human", "ai_atlas"}:
                game.keyboard.set_target(target)
                return f"keyboard control set to {target}"
            return "unknown control target (use human or ai_atlas)"
        if parts[:2] == ["teleport"] and len(parts) >= 4:
            target = parts[1]
            x, y = int(parts[2]), int(parts[3])
            actor = game.env.world.get_actor(target)
            actor.pos.x = x
            actor.pos.y = y
            return f"teleported {target}"
        if parts[:2] == ["pause", "ai"]:
            game.ai_paused = True
            return "AI paused"
        if parts[:2] == ["resume", "ai"]:
            game.ai_paused = False
            return "AI resumed"
        if parts[0] == "save":
            game.save()
            return "saved"
        if parts[0] == "load":
            game.load()
            return "loaded"
        if parts[:2] == ["reset", "episode"]:
            game.reset_episode()
            return "episode reset"
        if parts[:2] == ["print", "state"]:
            return str(game.env.world.atlas)
        if parts[:2] == ["print", "map"]:
            return str(game.env.world.tiles)
        if parts[:2] == ["agent", "info"]:
            return "Atlas RL Agent (RecurrentPPO)"
        return "unknown command"
