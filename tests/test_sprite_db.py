import pygame

from src.render.sprite_db import SpriteDB


def test_character_sprite_is_not_full_block_and_stays_in_bounds() -> None:
    pygame.init()
    surface = pygame.Surface((32, 32), pygame.SRCALPHA)

    SpriteDB(tile_size=32).draw_character(surface, (100, 160, 220), 0, 0)

    filled_pixels = []
    for y in range(32):
        for x in range(32):
            if surface.get_at((x, y)).a > 0:
                filled_pixels.append((x, y))

    assert filled_pixels, "sprite should draw visible pixels"
    xs = [p[0] for p in filled_pixels]
    ys = [p[1] for p in filled_pixels]

    assert min(xs) >= 0
    assert min(ys) >= 0
    assert max(xs) <= 31
    assert max(ys) <= 31

    # Ensure it is a silhouette rather than a full 32x32 block.
    assert len(filled_pixels) < 32 * 32
    assert surface.get_at((0, 0)).a == 0
    assert surface.get_at((31, 31)).a == 0
