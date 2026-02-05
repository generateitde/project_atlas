from src.agent.dagger import DAgger


def test_needs_query_requires_stuck_and_uncertainty() -> None:
    dagger = DAgger(uncertainty_threshold=0.5)
    assert not dagger.needs_query(stuck=False, uncertainty=0.9)
    assert not dagger.needs_query(stuck=True, uncertainty=0.2)
    assert dagger.needs_query(stuck=True, uncertainty=0.6)


def test_stuck_detection_from_loop_and_no_progress() -> None:
    dagger = DAgger(stuck_window=6, min_stuck_unique_positions=2, no_progress_limit=4)
    for pos in [(1, 1), (1, 2), (1, 1), (1, 2), (1, 1), (1, 2)]:
        stuck = dagger.update_stuck_state(pos, progress_signal=0.0)
    assert stuck

    dagger.reset()
    stuck = False
    for _ in range(4):
        stuck = dagger.update_stuck_state((5, 5), progress_signal=0.0)
    assert stuck


def test_entropy_normalized_range() -> None:
    value = DAgger.entropy_from_probs([0.25, 0.25, 0.25, 0.25])
    assert 0.99 <= value <= 1.01

    assert DAgger.entropy_from_probs([1.0, 0.0, 0.0, 0.0]) == 0.0
