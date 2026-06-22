from dcrh.core.boundaries import (
    DoubleNewlineBoundaryDetector,
    StatisticalBlockBuilder,
    TokenObservation,
)


def obs(i):
    return TokenObservation(
        token_id=i,
        entropy=0.5,
        grounding=0.2,
        text_piece="x",
        decode_seconds=0.0,
        probe_seconds=0.0,
        sequence_length_after_token=i,
    )


def test_newline_run_emits_once():
    detector = DoubleNewlineBoundaryDetector()
    assert detector.push("a\n", 1) is None
    boundary = detector.push("\n", 2)
    assert boundary is not None
    assert detector.push("\n", 3) is None
    assert detector.push("x", 4) is None
    assert detector.push("\n\n", 5) is not None


def test_token_mass_closes_at_natural_boundary():
    builder = StatisticalBlockBuilder(3, 0.9, 32)
    builder.add(obs(1))
    builder.add(obs(2))
    assert builder.on_atomic_boundary(2) is None
    builder.add(obs(3))
    block = builder.on_atomic_boundary(3)
    assert block is not None
    assert block.token_count == 3
    assert block.atomic_segments == 2


def test_token_mass_can_close_without_grounding_for_h_only_monitor():
    builder = StatisticalBlockBuilder(2, 0.9, 32, require_grounding=False)
    builder.add(
        TokenObservation(
            token_id=1,
            entropy=0.4,
            grounding=None,
            text_piece="x",
            decode_seconds=0.0,
            probe_seconds=0.0,
            sequence_length_after_token=1,
        )
    )
    builder.add(
        TokenObservation(
            token_id=2,
            entropy=0.6,
            grounding=None,
            text_piece="\n\n",
            decode_seconds=0.0,
            probe_seconds=0.0,
            sequence_length_after_token=2,
        )
    )
    block = builder.on_atomic_boundary(2)
    assert block is not None
    assert block.grounding is None
    assert block.entropy_mean == 0.5
