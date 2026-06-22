import json

from dcrh.config import DataConfig
from dcrh.evaluation.data import iter_examples
from dcrh.evaluation.verifier import verify


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_aime24_adapter_builds_prompt_and_gold(tmp_path):
    path = tmp_path / "aime24.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "aime24_001",
                "problem": "What is 20+22?",
                "answer": "42",
            }
        ],
    )

    example = next(iter_examples(DataConfig(path=str(path), dataset="aime24")))

    assert example.example_id == "aime24_001"
    assert "Solve the following math problem" in example.question
    assert "two newline characters" in example.question
    assert "Problem: What is 20+22?" in example.question
    assert example.answer == "42"


def test_gpqa_adapter_preserves_correct_answer_first_when_unlabeled(tmp_path):
    path = tmp_path / "gpqa.jsonl"
    write_jsonl(
        path,
        [
            {
                "question_id": "gpqa_1",
                "Question": "Which option is correct?",
                "Correct Answer": "correct",
                "Incorrect Answer 1": "wrong 1",
                "Incorrect Answer 2": "wrong 2",
                "Incorrect Answer 3": "wrong 3",
            }
        ],
    )

    example = next(iter_examples(DataConfig(path=str(path), dataset="gpqa")))

    assert example.example_id == "gpqa_1"
    assert "A. correct" in example.question
    assert "D. wrong 3" in example.question
    assert example.answer == "A"


def test_gpqa_adapter_uses_explicit_letter_gold(tmp_path):
    path = tmp_path / "gpqa_labeled.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "gpqa_2",
                "question": "Pick one.",
                "A": "alpha",
                "B": "beta",
                "C": "gamma",
                "D": "delta",
                "answer": "C",
            }
        ],
    )

    example = next(iter_examples(DataConfig(path=str(path), dataset="gpqa_diamond")))

    assert "C. gamma" in example.question
    assert example.answer == "C"


def test_auto_verifier_handles_aime_fraction_and_gpqa_choice():
    math_result = verify(
        output="After simplifying, the final answer is \\boxed{\\frac{2}{4}}.",
        gold="1/2",
        verifier="auto",
        dataset="aime25",
    )
    gpqa_result = verify(
        output="The final answer is \\boxed{B}.",
        gold="B",
        verifier="auto",
        dataset="gpqa",
    )

    assert math_result["correct"] is True
    assert gpqa_result["correct"] is True
