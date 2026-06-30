import json

from build_math_calibration import build_dataset


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_build_dataset_samples_by_difficulty(tmp_path):
    simple = tmp_path / "simple.jsonl"
    hard = tmp_path / "hard.jsonl"
    _write_jsonl(
        simple,
        [
            {"id": "s1", "problem": "1+1?", "answer": "2"},
            {"id": "s2", "problem": "2+2?", "answer": "4"},
        ],
    )
    _write_jsonl(
        hard,
        [
            {"id": "h1", "problem": "Prove x.", "answer": "x"},
            {"id": "h2", "problem": "Prove y.", "answer": "y"},
        ],
    )
    rows, summary = build_dataset(
        {
            "seed": 1,
            "prompt_style": "aime",
            "mix": {"simple": 1, "hard": 2},
            "sources": [
                {"name": "math500", "difficulty": "simple", "path": str(simple)},
                {"name": "olympiad", "difficulty": "hard", "path": str(hard)},
            ],
        }
    )
    assert len(rows) == 3
    assert summary["difficulty_counts"] == {"simple": 1, "hard": 2}
    assert all("\\boxed{}" in row["question"] for row in rows)


def test_build_dataset_deduplicates_problem_text(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [{"id": "a", "problem": "Same problem", "answer": "1"}])
    _write_jsonl(second, [{"id": "b", "problem": " same   problem ", "answer": "1"}])
    rows, summary = build_dataset(
        {
            "seed": 1,
            "sources": [
                {"name": "a", "difficulty": "simple", "path": str(first)},
                {"name": "b", "difficulty": "simple", "path": str(second)},
            ],
        }
    )
    assert len(rows) == 1
    assert summary["sample_count"] == 1
