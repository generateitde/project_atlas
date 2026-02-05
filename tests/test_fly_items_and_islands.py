from src.config import load_config
from src.core.rng import RNG
from src.core.types import TileType
from src.env.grid_env import GridEnv
from src.env.tools import precheck_move
from src.env.world_gen import floating_islands


def test_fly_override_allows_vertical_movement() -> None:
    env = GridEnv(load_config(), preset="floating_islands", seed=7)
    actor = env.world.atlas

    blocked = precheck_move(env.world, actor.entity_id, "N")
    assert blocked.ok is False
    assert blocked.error_code == "blocked"

    actor.can_fly = True
    allowed = precheck_move(env.world, actor.entity_id, "N")
    assert allowed.ok is True


def test_floating_islands_places_goal_on_high_island() -> None:
    width, height = 16, 12
    tiles = floating_islands(width, height, RNG(42))

    goal_positions = [(x, y) for y in range(height) for x in range(width) if tiles[y, x] == TileType.GOAL]
    assert goal_positions, "floating islands world must include a goal"
    goal_x, goal_y = goal_positions[0]

    assert goal_y <= 2
    assert tiles[goal_y + 1, goal_x] == TileType.PLATFORM
    ground_row = tiles[height - 2, 2 : width - 2]
    assert all(tile == TileType.EMPTY for tile in ground_row)
