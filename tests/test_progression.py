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


def test_transform_applies_stat_multipliers_and_timer() -> None:
    actor = Character(entity_id="ai_atlas", display_name="Atlas", pos=Vec2(1, 1), atk=4, defense=2, speed=1.0, jump_power=2.0)

    result = rules.activate_transform(
        actor,
        transform_id="berserk",
        duration_ticks=3,
        atk_multiplier=2.0,
        defense_multiplier=1.5,
        speed_multiplier=1.3,
        jump_multiplier=1.25,
    )

    assert result["transform"] == "berserk"
    assert actor.transform_state == "berserk"
    assert actor.transform_timer == 3
    assert actor.atk == 8
    assert actor.defense == 3
    assert actor.speed == 1.3
    assert actor.jump_power == 2.5


def test_transform_reverts_stats_after_duration() -> None:
    actor = Character(entity_id="ai_atlas", display_name="Atlas", pos=Vec2(1, 1), atk=6, defense=3, speed=1.5, jump_power=2.0)
    original = (actor.atk, actor.defense, actor.speed, actor.jump_power)

    rules.activate_transform(actor, duration_ticks=1, atk_multiplier=1.5, defense_multiplier=2.0, speed_multiplier=1.2, jump_multiplier=1.4)
    ended = rules.tick_transform(actor)

    assert ended is True
    assert actor.transform_state is None
    assert actor.transform_timer == 0
    assert actor.transform_stats_backup is None
    assert (actor.atk, actor.defense, actor.speed, actor.jump_power) == original
