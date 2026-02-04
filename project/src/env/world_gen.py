from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from src.core.rng import RNG
from src.core.types import TileType


@dataclass
class WorldData:
    tiles: List[List[TileType]]
    spawn_human: Tuple[int, int]
    spawn_atlas: Tuple[int, int]
    items: List[Tuple[str, int, int]]
    enemies: List[Tuple[str, int, int, int]]


PRESETS = [
    "floating_islands",
    "dungeon_exit",
    "arena_training",
    "ctf_small",
    "hide_seek_maze",
]


def _empty_grid(width: int, height: int) -> List[List[TileType]]:
    return [[TileType.EMPTY for _ in range(width)] for _ in range(height)]


def generate_world(preset: str, width: int, height: int, seed: int) -> WorldData:
    rng = RNG(seed)
    tiles = _empty_grid(width, height)
    if preset == "floating_islands":
        for x in range(width):
            tiles[height - 1][x] = TileType.WALL
        for _ in range(4):
            island_y = rng.randint(2, height - 4)
            island_x = rng.randint(2, width - 4)
            for x in range(island_x, min(width - 1, island_x + 4)):
                tiles[island_y][x] = TileType.PLATFORM
        items = [("fly_orb", width // 2, height // 2)]
        spawn_human = (1, height - 2)
        spawn_atlas = (3, height - 2)
    elif preset == "dungeon_exit":
        for x in range(width):
            tiles[0][x] = TileType.WALL
            tiles[height - 1][x] = TileType.WALL
        for y in range(height):
            tiles[y][0] = TileType.WALL
            tiles[y][width - 1] = TileType.WALL
        for x in range(2, width - 2):
            tiles[height // 2][x] = TileType.BREAKABLE_WALL
        tiles[height // 2][width // 2] = TileType.DOOR_CLOSED
        tiles[1][width - 2] = TileType.GOAL
        items = [("key_blue", 2, height - 2)]
        spawn_human = (1, height - 2)
        spawn_atlas = (3, height - 2)
    elif preset == "arena_training":
        for x in range(width):
            tiles[height - 1][x] = TileType.WALL
        tiles[height - 2][width // 2] = TileType.GATE
        items = []
        spawn_human = (1, height - 2)
        spawn_atlas = (width - 2, height - 2)
    elif preset == "ctf_small":
        for x in range(width):
            tiles[height - 1][x] = TileType.WALL
        tiles[height // 2][width // 2] = TileType.FLAG
        items = []
        spawn_human = (1, height - 2)
        spawn_atlas = (width - 2, height - 2)
    elif preset == "hide_seek_maze":
        for x in range(width):
            tiles[height - 1][x] = TileType.WALL
        for y in range(1, height - 1):
            tiles[y][2] = TileType.WALL
            tiles[y][width - 3] = TileType.WALL
        items = []
        spawn_human = (1, height - 2)
        spawn_atlas = (width - 2, height - 2)
    else:
        raise ValueError(f"Unknown preset: {preset}")

    enemies = [("slime", width // 2, height - 2, 5)]
    return WorldData(
        tiles=tiles,
        spawn_human=spawn_human,
        spawn_atlas=spawn_atlas,
        items=items,
        enemies=enemies,
    )
