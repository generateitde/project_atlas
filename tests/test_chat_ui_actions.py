from src.human.chat_ui import format_action_choices, parse_human_action_choice


def test_parse_human_action_by_index_and_alias() -> None:
    assert parse_human_action_choice("5") == 5
    assert parse_human_action_choice("jump") == 5
    assert parse_human_action_choice("move_e") == 2


def test_parse_respects_action_mask() -> None:
    action_mask = [False] * 14
    action_mask[5] = True
    assert parse_human_action_choice("jump", action_mask) == 5
    assert parse_human_action_choice("2", action_mask) is None


def test_format_action_choices_uses_mask() -> None:
    action_mask = [False] * 14
    action_mask[0] = True
    action_mask[5] = True
    text = format_action_choices(action_mask)
    assert "0:NOOP" in text
    assert "5:JUMP" in text
    assert "2:MOVE_E" not in text
