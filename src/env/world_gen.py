from __future__ import annotations

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
    tiles[height - 2, width - 2] = TileType.GOAL
    tiles[height - 2, 2] = TileType.DOOR_CLOSED
    tiles[height - 3, 2] = TileType.FLAG
    for x in range(2, width - 2):
        tiles[height - 4, x] = TileType.BREAKABLE_WALL
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
