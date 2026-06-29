from dcrh.core.boundaries import DoubleNewlineBoundaryDetector


def test_newline_run_emits_once():
    detector = DoubleNewlineBoundaryDetector()
    assert detector.push("a\n", 1) is None
    boundary = detector.push("\n", 2)
    assert boundary is not None
    assert boundary.token_index == 2
    assert boundary.atomic_length == 2
    assert detector.push("\n", 3) is None
    assert detector.push("x", 4) is None
    assert detector.push("\n\n", 5) is not None


def test_delimiter_belongs_to_current_chunk():
    detector = DoubleNewlineBoundaryDetector()
    pieces = ["Step", " one", ".\n", "\n", "Next"]
    boundary = None
    for index, piece in enumerate(pieces, start=1):
        boundary = detector.push(piece, index) or boundary
    assert boundary is not None
    assert boundary.token_index == 4
