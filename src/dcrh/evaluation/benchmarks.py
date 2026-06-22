from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MATH_BENCHMARKS = {"aime24", "aime25"}
GPQA_BENCHMARKS = {"gpqa", "gpqa_diamond"}
SUPPORTED_BENCHMARKS = MATH_BENCHMARKS | GPQA_BENCHMARKS


@dataclass(slots=True)
class BenchmarkExample:
    example_id: str
    question: str
    answer: str | None


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value).strip()


def _canonical_dataset(dataset: str) -> str:
    name = dataset.lower()
    if name not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark dataset: {dataset}")
    return name


def _default_id(row: dict[str, Any], dataset: str, index: int, configured_id: str | None) -> str:
    raw = configured_id
    if raw is None:
        raw = _first_present(row, ["id", "task_id", "question_id", "problem_id"])
    if raw is None:
        return f"{dataset}_{index:04d}"
    raw_text = str(raw)
    if raw_text.startswith("HumanEval/"):
        raw_text = raw_text.rsplit("/", 1)[-1]
    return raw_text


def _math_prompt(row: dict[str, Any], dataset: str) -> str:
    body = _first_present(row, ["problem", "question", "prompt"])
    if body is None:
        raise KeyError(
            f"{dataset} row must contain one of: problem, question, prompt"
        )
    return (
        "Solve the following math problem and return ONLY the final answer.\n"
        "Please reason step by step, separate logical reasoning steps with two newline "
        "characters (\\n\\n), and put your final answer within \\boxed{}.\n\n"
        f"Problem: {body}\n\n"
    )


def _math_gold(row: dict[str, Any]) -> str | None:
    return _string_or_none(_first_present(row, ["answer", "solution", "target"]))


def _choice_value(row: dict[str, Any], label: str) -> Any:
    return _first_present(
        row,
        [
            label,
            label.lower(),
            f"choice_{label.lower()}",
            f"choice_{label}",
            f"Choice {label}",
        ],
    )


def _gpqa_choices(row: dict[str, Any]) -> list[tuple[str, str]]:
    explicit = [
        (label, str(value).strip())
        for label in ["A", "B", "C", "D"]
        for value in [_choice_value(row, label)]
        if value is not None and str(value).strip()
    ]
    if explicit:
        return explicit

    ordered_values = [
        row.get("Correct Answer"),
        row.get("Incorrect Answer 1"),
        row.get("Incorrect Answer 2"),
        row.get("Incorrect Answer 3"),
    ]
    return [
        (label, str(value).strip())
        for label, value in zip(["A", "B", "C", "D"], ordered_values)
        if value is not None and str(value).strip()
    ]


def _gpqa_prompt(row: dict[str, Any], dataset: str) -> str:
    body = _first_present(row, ["problem", "Question", "question", "prompt"])
    if body is None:
        raise KeyError(
            f"{dataset} row must contain one of: problem, Question, question, prompt"
        )
    choices = _gpqa_choices(row)
    if choices:
        choices_text = "\n".join(f"{label}. {value}" for label, value in choices)
        body = f"{body}\n\n{choices_text}"
    return (
        "What is the correct answer to the following problem? Please reason step by step.\n"
        "Separate logical reasoning steps with two newline characters (\\n\\n).\n"
        "Put the final answer strictly in the format \\boxed{X}, where X is a single "
        "letter (A, B, C, or D).\n\n"
        f"Problem: {body}\n\n"
    )


def _gpqa_gold(row: dict[str, Any]) -> str | None:
    for key in ["answer", "correct_answer", "label", "target"]:
        value = _string_or_none(row.get(key))
        if value:
            candidate = value.strip().upper()
            if candidate[:1] in {"A", "B", "C", "D"}:
                return candidate[:1]
            choices = _gpqa_choices(row)
            for label, choice in choices:
                if choice.strip().lower() == value.strip().lower():
                    return label
            return value

    correct_text = _string_or_none(row.get("Correct Answer"))
    if correct_text is None:
        return None
    choices = _gpqa_choices(row)
    for label, choice in choices:
        if choice.strip().lower() == correct_text.strip().lower():
            return label
    # GPQA-Diamond rows commonly provide Correct Answer plus three incorrect
    # answers, and the prompt above preserves that order.
    return "A"


def build_benchmark_example(
    row: dict[str, Any],
    dataset: str,
    index: int,
    configured_id: str | None = None,
) -> BenchmarkExample:
    dataset = _canonical_dataset(dataset)
    if dataset in MATH_BENCHMARKS:
        return BenchmarkExample(
            example_id=_default_id(row, dataset, index, configured_id),
            question=_math_prompt(row, dataset),
            answer=_math_gold(row),
        )
    return BenchmarkExample(
        example_id=_default_id(row, dataset, index, configured_id),
        question=_gpqa_prompt(row, dataset),
        answer=_gpqa_gold(row),
    )
