from src.core.rng import RNG
from src.core.types import Character, TileType, Vec2
from src.env.grid_env import World
from src.env.modes import CaptureTheFlag
from src.env.world_gen import ctf_small


def _find_first_tile(tiles, tile_type: TileType) -> tuple[int, int]:
    for y in range(tiles.shape[0]):
        for x in range(tiles.shape[1]):
            if tiles[y, x] == tile_type:
                return x, y
    raise AssertionError(f"tile {tile_type} not found")


def test_capture_the_flag_pickup_tick_score_reset() -> None:
    width, height = 20, 12
    tiles = ctf_small(width, height, RNG(11))
    atlas = Character(entity_id="ai_atlas", display_name="Atlas", pos=Vec2(2, height - 2))
    human = Character(entity_id="human", display_name="Human", pos=Vec2(3, height - 2))
    world = World(tiles=tiles, atlas=atlas, human=human)

    mode = CaptureTheFlag()
    info = mode.reset(world, RNG(11)).model_dump()
    score_zone = info["score_zone"]
    assert score_zone == (2, height - 2)

    flag_x, flag_y = _find_first_tile(tiles, TileType.FLAG)
    world.atlas.pos = Vec2(flag_x, flag_y)
    reward_pickup, events_pickup, done_pickup, info_pickup = mode.step(world, [], RNG(11))

    assert reward_pickup >= 1.1
    assert done_pickup is False
    assert info_pickup["atlas_has_flag"] is True
    assert any(event.type == "ctf_flag_pickup" for event in events_pickup)

    world.atlas.pos = Vec2(*score_zone)
    reward_score, events_score, done_score, info_score = mode.step(world, [], RNG(11))

    assert reward_score >= 5.1
    assert done_score is True
    assert info_score["atlas_has_flag"] is False
    assert info_score["score"] == 1
    assert any(event.type == "ctf_scored" for event in events_score)
