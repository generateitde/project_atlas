from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from src.config import AtlasConfig
from src.core.events import Event
from src.core.rng import RNG
from src.core.types import TILE_PROPS, Character, Facing, TileType, Vec2
from src.env import encoding
from src.env.modes import Mode, create_mode
from src.env.rewards import compute_reward
from src.env.tools import ask_human, break_tile, inspect, jump, move, speak
from src.env.world_gen import default_spawn, generate_world


@dataclass
class World:
    tiles: np.ndarray
    atlas: Character
    human: Character
    items: list = field(default_factory=list)
    enemies: list = field(default_factory=list)
    messages: list[tuple[str, str]] = field(default_factory=list)
    atlas_has_flag: bool = False
    hand_item: Any | None = None
    pending_question: bool = False

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.tiles.shape[1] and 0 <= y < self.tiles.shape[0]

    def tile_at(self, pos: Vec2) -> TileType:
        x, y = int(pos.x), int(pos.y)
        if not self.in_bounds((x, y)):
            return TileType.WALL
        return self.tiles[y, x]

    def is_passable(self, pos: tuple[int, int]) -> bool:
        if not self.in_bounds(pos):
            return False
        tile = self.tiles[pos[1], pos[0]]
        return tile in {TileType.EMPTY, TileType.GOAL, TileType.FLAG, TileType.DOOR_OPEN, TileType.PLATFORM, TileType.LADDER}

    def can_stand_on(self, pos: tuple[int, int]) -> bool:
        if not self.in_bounds(pos):
            return False
        props = TILE_PROPS[self.tiles[pos[1], pos[0]]]
        return props.solid or props.one_way_platform

    def get_actor(self, actor_id: str) -> Character:
        if actor_id == self.atlas.entity_id:
            return self.atlas
        return self.human

    def describe_at(self, pos: tuple[int, int]) -> str:
        x, y = pos
        if not self.in_bounds(pos):
            return f"({x}, {y}) is outside the world bounds."
        tile = self.tiles[y, x]
        actors = []
        if (int(self.atlas.pos.x), int(self.atlas.pos.y)) == (x, y):
            actors.append(f"Atlas (HP {self.atlas.hp}, Lvl {self.atlas.level})")
        if (int(self.human.pos.x), int(self.human.pos.y)) == (x, y):
            actors.append("Human")
        actor_text = f" Actors: {', '.join(actors)}." if actors else ""
        return f"Tile ({x}, {y}): {tile.value}.{actor_text}"


class GridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: AtlasConfig, preset: str = "floating_islands", seed: int | None = None):
        super().__init__()
        self.config = config
        self.preset = preset
        self.seed_value = seed or config.training.seed
        self.rng = RNG(self.seed_value)
        self.mode: Mode = create_mode("ExitGame")
        self.world = self._build_world()
        radius = config.world.visibility_radius
        tile_shape = (2 * radius + 1, 2 * radius + 1)
        self.observation_space = gym.spaces.Dict(
            {
                "local_tiles": gym.spaces.Box(low=0, high=len(TileType), shape=tile_shape, dtype=np.int32),
                "local_entities": gym.spaces.Box(low=-10, high=10, shape=(4, 3), dtype=np.float32),
                "hand_item": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
                "stats": gym.spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32),
                "mode_features": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(encoding.ACTION_COUNT,), dtype=np.int8),
                "memory_hint": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Discrete(encoding.ACTION_COUNT)
        self._steps = 0

    def _build_world(self) -> World:
        tiles = generate_world(self.preset, self.config.world.width, self.config.world.height, self.rng)
        atlas = Character(entity_id="ai_atlas", display_name="Atlas", pos=default_spawn(self.config.world.width, self.config.world.height))
        human = Character(entity_id="human", display_name="Human", pos=Vec2(atlas.pos.x + 1, atlas.pos.y))
        return World(tiles=tiles, atlas=atlas, human=human)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = seed
        self.rng = RNG(self.seed_value)
        self.world = self._build_world()
        self._steps = 0
        self.mode.reset(self.world, self.rng)
        return self._obs(), {}

    def set_mode(self, name: str, params: dict | None = None) -> None:
        self.mode = create_mode(name, params)

    def step(self, action: int):
        events: list[Event] = []
        atlas = self.world.atlas
        if action == 2:
            move(self.world, atlas.entity_id, "E")
        elif action == 4:
            move(self.world, atlas.entity_id, "W")
        elif action == 5:
            jump(self.world, atlas.entity_id)
        elif action == 10:
            dx, dy = encoding.facing_to_dir(atlas.facing)
            break_tile(self.world, atlas.entity_id, int(atlas.pos.x + dx), int(atlas.pos.y + dy))
        elif action == 11:
            dx, dy = encoding.facing_to_dir(atlas.facing)
            inspect(self.world, atlas.entity_id, int(atlas.pos.x + dx), int(atlas.pos.y + dy))
        elif action == 12:
            speak(self.world, "Atlas (Plan): Weiter erkunden.")
        elif action == 13:
            ask_human(self.world, "Was soll ich als Nächstes tun?")

        _apply_vertical_motion(self.world, atlas)
        _apply_vertical_motion(self.world, self.world.human)

        mode_reward, mode_events, done, info = self.mode.step(self.world, events, self.rng)
        reward = compute_reward(mode_reward, events + mode_events)
        self._steps += 1
        if self._steps >= self.config.world.max_episode_steps:
            done = True
        return self._obs(), reward, done, False, info

    def _obs(self) -> dict[str, Any]:
        atlas = self.world.atlas
        radius = self.config.world.visibility_radius
        tiles = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = int(atlas.pos.x + dx)
                y = int(atlas.pos.y + dy)
                if self.world.in_bounds((x, y)):
                    tiles[dy + radius, dx + radius] = list(TileType).index(self.world.tiles[y, x])
        entities = np.zeros((4, 3), dtype=np.float32)
        entities[0] = np.array([1, self.world.human.pos.x - atlas.pos.x, self.world.human.pos.y - atlas.pos.y])
        entities[1] = np.array([2, 0.0, 0.0])
        hand = np.zeros((4,), dtype=np.float32)
        stats = np.array([atlas.hp, atlas.level, atlas.exp, atlas.speed], dtype=np.float32)
        mode_features = np.array([1.0, 0.0], dtype=np.float32)
        action_mask = encoding.action_mask_for(self.world, atlas).astype(np.int8)
        memory_hint = np.array([0.0], dtype=np.float32)
        return {
            "local_tiles": tiles,
            "local_entities": entities,
            "hand_item": hand,
            "stats": stats,
            "mode_features": mode_features,
            "action_mask": action_mask,
            "memory_hint": memory_hint,
        }
