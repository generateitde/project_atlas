from src.core.rng import RNG
from src.core.types import Character, Vec2
from src.env.modes import ExitGame
from src.env.world_gen import check_solvable, dungeon_exit
from src.env.grid_env import World


def test_dungeon_exit_solvable_for_100_seeds() -> None:
    width, height = 20, 12
    solvable = sum(check_solvable("dungeon_exit", width, height, seed) for seed in range(100))
    assert solvable >= 95


def test_exit_game_collect_key_opens_door() -> None:
    width, height = 20, 12
    tiles = dungeon_exit(width, height, RNG(7))
    atlas = Character(entity_id="ai_atlas", display_name="Atlas", pos=Vec2(2, height - 2))
    human = Character(entity_id="human", display_name="Human", pos=Vec2(3, height - 2))
    world = World(tiles=tiles, atlas=atlas, human=human)

    mode = ExitGame()
    mode.reset(world, RNG(7))

    key_positions = [(x, y) for y in range(height) for x in range(width) if tiles[y, x].value == "FLAG"]
    assert key_positions, "Expected a key tile (FLAG) in dungeon_exit preset"
    key_x, key_y = key_positions[0]
    world.atlas.pos = Vec2(key_x, key_y)

    reward, _, done, info = mode.step(world, [], RNG(7))

    assert reward >= 1.0
    assert not done
    assert info["key_collected"] is True
    assert any(
        world.tiles[y, x].value == "DOOR_OPEN"
        for y in range(height)
        for x in range(width)
    )
