from src.core.events import Event
from src.core.types import Character, TileType, Vec2
from src.env.grid_env import World
from src.env import rules


def test_gate_requires_level_for_passability() -> None:
    actor = Character(entity_id="ai_atlas", display_name="Atlas", pos=Vec2(1, 1), level=1)
    assert rules.can_pass_tile(actor, TileType.GATE) is True

    actor.level = 0
    assert rules.can_pass_tile(actor, TileType.GATE) is False


def test_exp_grant_levels_up_actor() -> None:
    actor = Character(entity_id="ai_atlas", display_name="Atlas", pos=Vec2(1, 1), level=1, exp=0)
    result = rules.grant_exp(actor, 10)

    assert result["levels_gained"] == 1
    assert actor.level == 2
    assert actor.exp == 0


def test_objective_events_map_to_exp() -> None:
    exp = rules.objective_exp_from(0.0, [Event("ctf_scored", {}), Event("unknown", {})])
    assert exp == 12


def test_world_switch_keeps_progression_levels() -> None:
    import numpy as np

    tiles = np.array([[TileType.EMPTY, TileType.GATE], [TileType.WALL, TileType.WALL]], dtype=object)
    atlas = Character(entity_id="ai_atlas", display_name="Atlas", pos=Vec2(0, 0), level=2)
    human = Character(entity_id="human", display_name="Human", pos=Vec2(0, 0))
    world = World(tiles=tiles, atlas=atlas, human=human)
    assert world.is_passable_for(atlas, (1, 0)) is True
