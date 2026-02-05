from src.console import Console
from src.core.rng import RNG
from src.core.types import Character, Vec2
from src.env.grid_env import World
from src.env.modes import HideAndSeek
from src.env.world_gen import hide_seek_maze


def test_hide_seek_distance_shaping_and_find_bonus() -> None:
    width, height = 20, 12
    tiles = hide_seek_maze(width, height, RNG(21))
    atlas = Character(entity_id="ai_atlas", display_name="Atlas", pos=Vec2(2, height - 2))
    human = Character(entity_id="human", display_name="Human", pos=Vec2(3, height - 2))
    world = World(tiles=tiles, atlas=atlas, human=human)

    mode = HideAndSeek(hide_target=(5, height - 2))
    mode.reset(world, RNG(21))

    world.atlas.pos = Vec2(3, height - 2)
    reward_closer, _, done_closer, info_closer = mode.step(world, [], RNG(21))
    assert reward_closer > 0
    assert done_closer is False
    assert info_closer["details"]["distance"] == 2

    world.atlas.pos = Vec2(5, height - 2)
    reward_found, events_found, done_found, info_found = mode.step(world, [], RNG(21))
    assert reward_found >= 5.0
    assert done_found is True
    assert info_found["status"] == "Target found!"
    assert any(event.type == "hide_target_found" for event in events_found)


def test_hide_seek_time_limit_ends_episode() -> None:
    width, height = 20, 12
    tiles = hide_seek_maze(width, height, RNG(22))
    atlas = Character(entity_id="ai_atlas", display_name="Atlas", pos=Vec2(2, height - 2))
    human = Character(entity_id="human", display_name="Human", pos=Vec2(3, height - 2))
    world = World(tiles=tiles, atlas=atlas, human=human)

    mode = HideAndSeek(hide_target=(12, 2), time_limit_steps=2)
    mode.reset(world, RNG(22))
    _, _, done_1, _ = mode.step(world, [], RNG(22))
    _, events_2, done_2, info_2 = mode.step(world, [], RNG(22))

    assert done_1 is False
    assert done_2 is True
    assert info_2["status"] == "Time limit reached."
    assert any(event.type == "hide_time_limit" for event in events_2)


def test_console_hide_set_marker_updates_hide_seek_target() -> None:
    class FakeGame:
        def __init__(self) -> None:
            self.env = type("Env", (), {})()
            self.env.world = type("WorldStub", (), {})()
            self.env.world.in_bounds = lambda pos: 0 <= pos[0] < 20 and 0 <= pos[1] < 12
            self.env.world.human = Character(entity_id="human", display_name="Human", pos=Vec2(7, 8))
            self.env.world.atlas = Character(entity_id="ai_atlas", display_name="Atlas", pos=Vec2(2, 2))
            self.env.mode = HideAndSeek()
            self.env.rng = RNG(1)

    game = FakeGame()
    console = Console()

    msg = console.execute(game, "hide set 9 4")
    assert msg == "hide marker set to (9, 4)"
    assert game.env.mode.hide_target == (9, 4)

    msg_default_y = console.execute(game, "hide set 10")
    assert msg_default_y == "hide marker set to (10, 8)"
    assert game.env.mode.hide_target == (10, 8)
