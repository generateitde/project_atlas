from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.core.rng import RNG
from src.core.types import TileType, Vec2


@dataclass
class WorldPreset:
    name: str
    generator: Callable[[int, int, RNG], np.ndarray]


def _blank(width: int, height: int, rng: RNG) -> np.ndarray:
    tiles = np.empty((height, width), dtype=object)
    tiles[:] = TileType.EMPTY
    tiles[0, :] = TileType.WALL
    tiles[-1, :] = TileType.WALL
    tiles[:, 0] = TileType.WALL
    tiles[:, -1] = TileType.WALL
    return tiles


def floating_islands(width: int, height: int, rng: RNG) -> np.ndarray:
    tiles = _blank(width, height, rng)
    for _ in range(6):
        cx = rng.randint(2, width - 3)
        cy = rng.randint(2, height - 4)
        tiles[cy, cx - 1 : cx + 2] = TileType.PLATFORM
    tiles[height - 2, width - 2] = TileType.GOAL
    return tiles


def dungeon_exit(width: int, height: int, rng: RNG) -> np.ndarray:
    tiles = _blank(width, height, rng)
    corridor_y = height - 2
    spawn_x = 2
    key_x = max(3, rng.randint(3, max(4, width // 2)))
    door_x = min(width - 4, max(key_x + 2, rng.randint(max(5, width // 2), width - 4)))
    goal_x = width - 2

    # Guaranteed solvable baseline path on the corridor row:
    # spawn -> key(FLAG) -> door(DOOR_CLOSED, opens with key) -> goal.
    tiles[corridor_y, key_x] = TileType.FLAG
    tiles[corridor_y, door_x] = TileType.DOOR_CLOSED
    tiles[corridor_y, goal_x] = TileType.GOAL

    # Optional breakable shortcuts/hazards above the corridor that do not block solvability.
    for x in range(2, width - 2):
        if x not in {key_x, door_x} and rng.random() < 0.28:
            tiles[corridor_y - 1, x] = TileType.BREAKABLE_WALL

    tiles[corridor_y, spawn_x] = TileType.EMPTY
    return tiles


def arena_training(width: int, height: int, rng: RNG) -> np.ndarray:
    tiles = _blank(width, height, rng)
    tiles[height - 2, width // 2] = TileType.FLAG
    return tiles


def ctf_small(width: int, height: int, rng: RNG) -> np.ndarray:
    tiles = _blank(width, height, rng)
    tiles[2, 2] = TileType.FLAG
    tiles[height - 3, width - 3] = TileType.FLAG
    return tiles


def hide_seek_maze(width: int, height: int, rng: RNG) -> np.ndarray:
    tiles = _blank(width, height, rng)
    for y in range(2, height - 2, 2):
        for x in range(2, width - 2, 2):
            tiles[y, x] = TileType.WALL
    return tiles


PRESETS: dict[str, WorldPreset] = {
    "floating_islands": WorldPreset("floating_islands", floating_islands),
    "dungeon_exit": WorldPreset("dungeon_exit", dungeon_exit),
    "arena_training": WorldPreset("arena_training", arena_training),
    "ctf_small": WorldPreset("ctf_small", ctf_small),
    "hide_seek_maze": WorldPreset("hide_seek_maze", hide_seek_maze),
}


def generate_world(preset: str, width: int, height: int, rng: RNG) -> np.ndarray:
    if preset not in PRESETS:
        preset = "floating_islands"
    return PRESETS[preset].generator(width, height, rng)


def default_spawn(width: int, height: int) -> Vec2:
    return Vec2(2, height - 2)


def world_snapshot_hash(tiles: np.ndarray) -> str:
    flattened = ",".join(tile.value for tile in tiles.ravel())
    payload = f"{tiles.shape[0]}x{tiles.shape[1]}|{flattened}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_solvable(preset: str, width: int, height: int, seed: int) -> bool:
    rng = RNG(seed)
    tiles = generate_world(preset, width, height, rng)
    spawn = default_spawn(width, height).as_int()
    goals = {(x, y) for y in range(height) for x in range(width) if tiles[y, x] == TileType.GOAL}
    if not goals:
        return False

    frontier = [(spawn[0], spawn[1], False)]
    visited: set[tuple[int, int, bool]] = {(spawn[0], spawn[1], False)}

    def passable(tile: TileType, has_key: bool) -> bool:
        if tile in {TileType.WALL, TileType.BREAKABLE_WALL, TileType.WATER, TileType.LAVA, TileType.GATE}:
            return False
        if tile == TileType.DOOR_CLOSED:
            return has_key
        return True

    while frontier:
        x, y, has_key = frontier.pop(0)
        if (x, y) in goals:
            return True
        current_has_key = has_key or tiles[y, x] == TileType.FLAG
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            tile = tiles[ny, nx]
            if not passable(tile, current_has_key):
                continue
            state = (nx, ny, current_has_key)
            if state in visited:
                continue
            visited.add(state)
            frontier.append(state)
    return False
