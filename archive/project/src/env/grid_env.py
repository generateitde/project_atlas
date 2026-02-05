from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.core.events import Event
from src.core.rng import RNG
from src.core.types import Character, Enemy, Facing, Item, ItemProps, TileType
from src.env.encoding import encode_tile
from src.env.modes import MODE_REGISTRY, Mode
from src.env.rewards import RewardTracker
from src.env.rules import TILE_PROPS, is_passable
from src.env.world_gen import PRESETS, WorldData, generate_world

Action = int


@dataclass
class WorldState:
    tiles: List[List[TileType]]
    human: Character
    atlas: Character
    items: Dict[str, Item]
    enemies: Dict[str, Enemy]
    tick: int = 0
    hand_flag: Optional[str] = None


class GridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, width: int, height: int, visibility_radius: int, preset: str, seed: int):
        super().__init__()
        self.width = width
        self.height = height
        self.visibility_radius = visibility_radius
        self.preset = preset
        self.seed = seed
        self.rng = RNG(seed)
        self.mode: Mode = MODE_REGISTRY["FreeExplore"]()
        self.reward_tracker = RewardTracker()
        self._world = self._build_world()
        self.action_space = gym.spaces.Discrete(14)
        self.observation_space = gym.spaces.Dict(
            {
                "local_tiles": gym.spaces.Box(
                    low=0,
                    high=len(TileType),
                    shape=(2 * visibility_radius + 1, 2 * visibility_radius + 1),
                    dtype=np.int32,
                ),
                "local_entities": gym.spaces.Box(low=-1, high=10, shape=(6, 6), dtype=np.float32),
                "hand_item": gym.spaces.Box(low=0, high=255, shape=(4,), dtype=np.float32),
                "stats": gym.spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32),
                "mode_features": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32),
                "memory_hint": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )

    @property
    def world(self) -> WorldState:
        return self._world

    def switch_world(self, preset: str, seed: int) -> None:
        self.preset = preset
        self.seed = seed
        self.rng.seed(seed)
        self._world = self._build_world()

    def set_mode(self, mode_name: str) -> None:
        if mode_name not in MODE_REGISTRY:
            raise ValueError(f"Unknown mode {mode_name}")
        self.mode = MODE_REGISTRY[mode_name]()
        self.mode.reset(self._world, self.rng)

    def _build_world(self) -> WorldState:
        data: WorldData = generate_world(self.preset, self.width, self.height, self.seed)
        human = Character(entity_id="human", display_name="Human", pos=data.spawn_human)
        atlas = Character(entity_id="ai_atlas", display_name="Atlas", pos=data.spawn_atlas)
        items = {}
        for idx, (item_type, x, y) in enumerate(data.items):
            props = ItemProps(
                pickupable=True,
                key_id="blue" if "key" in item_type else None,
                fly_grant={"mode": "temp", "duration": 50, "speed": 2} if "fly" in item_type else None,
            )
            items[f"item_{idx}"] = Item(item_id=f"item_{idx}", pos=(x, y), item_type=item_type, props=props)
        enemies = {}
        for idx, (etype, x, y, hp) in enumerate(data.enemies):
            enemies[f"enemy_{idx}"] = Enemy(f"enemy_{idx}", pos=(x, y), hp=hp, atk=1, exp_value=5)
        return WorldState(tiles=data.tiles, human=human, atlas=atlas, items=items, enemies=enemies)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed = seed
        self._world = self._build_world()
        self.reward_tracker = RewardTracker()
        obs = self._observe(self._world.atlas)
        return obs, {}

    def _observe(self, actor: Character) -> Dict[str, np.ndarray]:
        radius = self.visibility_radius
        local_tiles = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = actor.pos[0] + dx
                y = actor.pos[1] + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    local_tiles[dy + radius, dx + radius] = encode_tile(self._world.tiles[y][x])
        local_entities = np.zeros((6, 6), dtype=np.float32)
        local_entities[0, :2] = actor.pos
        local_entities[1, :2] = self._world.human.pos
        for idx, enemy in enumerate(self._world.enemies.values()):
            if idx + 2 < 6:
                local_entities[idx + 2, :2] = enemy.pos
        hand_item = np.zeros(4, dtype=np.float32)
        if actor.hand_item_id and actor.hand_item_id in self._world.items:
            hand_item[0] = 1
        stats = np.array(
            [
                actor.stats.hp,
                actor.stats.atk,
                actor.stats.defense,
                actor.stats.level,
                actor.stats.exp,
                float(actor.stats.can_fly),
            ],
            dtype=np.float32,
        )
        mode_features = np.zeros(4, dtype=np.float32)
        action_mask = self._action_mask(actor)
        memory_hint = np.array([0.0], dtype=np.float32)
        return {
            "local_tiles": local_tiles,
            "local_entities": local_entities,
            "hand_item": hand_item,
            "stats": stats,
            "mode_features": mode_features,
            "action_mask": action_mask,
            "memory_hint": memory_hint,
        }

    def _action_mask(self, actor: Character) -> np.ndarray:
        mask = np.ones(14, dtype=np.float32)
        if actor.hand_item_id is None:
            mask[7] = 0
            mask[8] = 0
        return mask

    def step(self, action: Action):
        events: List[Event] = []
        reward = 0.0
        done = False
        info: Dict[str, float] = {}
        actor = self._world.atlas
        if action == 1:
            self._move(actor, Facing.N, events)
        elif action == 2:
            self._move(actor, Facing.E, events)
        elif action == 3:
            self._move(actor, Facing.S, events)
        elif action == 4:
            self._move(actor, Facing.W, events)
        elif action == 5:
            self._jump(actor, events)
        elif action == 6:
            self._pickup(actor, events)
        elif action == 7:
            self._drop(actor, events)
        elif action == 8:
            self._use(actor, events)
        elif action == 9:
            self._attack(actor, events)
        elif action == 10:
            self._break(actor, events)
        elif action == 11:
            self._inspect(actor, events)
        elif action == 12:
            events.append(Event("atlas_speak", {"text": "Atlas (Plan): Explore."}))
        elif action == 13:
            events.append(Event("ask_human", {"text": "Atlas (Frage): Welche Aktion?"}))

        self._world.tick += 1
        mode_reward, mode_events, mode_done, mode_info = self.mode.step(self._world, events, self.rng)
        reward += mode_reward
        reward += self.reward_tracker.progress_reward(events)
        reward += self.reward_tracker.exploration_reward(self._visible_tiles(actor))
        info.update(mode_info)
        done = mode_done
        obs = self._observe(actor)
        return obs, reward, done, False, info

    def _visible_tiles(self, actor: Character) -> List[Tuple[int, int]]:
        tiles = []
        for dy in range(-self.visibility_radius, self.visibility_radius + 1):
            for dx in range(-self.visibility_radius, self.visibility_radius + 1):
                x = actor.pos[0] + dx
                y = actor.pos[1] + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    tiles.append((x, y))
        return tiles

    def _move(self, actor: Character, facing: Facing, events: List[Event]) -> None:
        actor.facing = facing
        dx, dy = 0, 0
        if facing == Facing.N:
            dy = -1
        elif facing == Facing.S:
            dy = 1
        elif facing == Facing.E:
            dx = 1
        elif facing == Facing.W:
            dx = -1
        nx = actor.pos[0] + dx
        ny = actor.pos[1] + dy
        if 0 <= nx < self.width and 0 <= ny < self.height:
            tile = self._world.tiles[ny][nx]
            if is_passable(tile, actor.stats.level):
                actor.pos = (nx, ny)
                if tile == TileType.GOAL:
                    events.append(Event("goal_reached", {"actor_id": actor.entity_id}))
                if tile == TileType.FLAG:
                    events.append(Event("flag_carried", {"actor_id": actor.entity_id}))
        events.append(Event("move", {"actor_id": actor.entity_id, "pos": actor.pos}))

    def _jump(self, actor: Character, events: List[Event]) -> None:
        actor.stats.jump_power += 0.1
        events.append(Event("jump", {"actor_id": actor.entity_id}))

    def _pickup(self, actor: Character, events: List[Event]) -> None:
        for item in self._world.items.values():
            if item.pos == actor.pos and item.props.pickupable:
                actor.hand_item_id = item.item_id
                events.append(Event("pickup", {"actor_id": actor.entity_id, "item_id": item.item_id}))
                return

    def _drop(self, actor: Character, events: List[Event]) -> None:
        if actor.hand_item_id:
            item = self._world.items.get(actor.hand_item_id)
            if item:
                item.pos = actor.pos
            events.append(Event("drop", {"actor_id": actor.entity_id, "item_id": actor.hand_item_id}))
            actor.hand_item_id = None

    def _use(self, actor: Character, events: List[Event]) -> None:
        if not actor.hand_item_id:
            return
        item = self._world.items.get(actor.hand_item_id)
        if item and item.props.key_id:
            for y in range(self.height):
                for x in range(self.width):
                    if self._world.tiles[y][x] == TileType.DOOR_CLOSED:
                        self._world.tiles[y][x] = TileType.DOOR_OPEN
                        events.append(Event("door_open", {"x": x, "y": y}))
                        return
        if item and item.props.fly_grant:
            actor.stats.can_fly = True
            events.append(Event("fly_granted", {"actor_id": actor.entity_id}))

    def _attack(self, actor: Character, events: List[Event]) -> None:
        for enemy in self._world.enemies.values():
            if enemy.pos == actor.pos:
                enemy.hp -= actor.stats.atk
                events.append(Event("attack", {"actor_id": actor.entity_id, "enemy_id": enemy.enemy_id}))
                if enemy.hp <= 0:
                    events.append(Event("enemy_defeated", {"enemy_id": enemy.enemy_id}))
                return

    def _break(self, actor: Character, events: List[Event]) -> None:
        x, y = actor.pos
        if actor.facing == Facing.N:
            y -= 1
        elif actor.facing == Facing.S:
            y += 1
        elif actor.facing == Facing.E:
            x += 1
        elif actor.facing == Facing.W:
            x -= 1
        if 0 <= x < self.width and 0 <= y < self.height:
            tile = self._world.tiles[y][x]
            if TILE_PROPS[tile].breakable:
                self._world.tiles[y][x] = TileType.EMPTY
                events.append(Event("tile_broken", {"x": x, "y": y}))

    def _inspect(self, actor: Character, events: List[Event]) -> None:
        events.append(Event("inspect", {"actor_id": actor.entity_id, "pos": actor.pos}))
