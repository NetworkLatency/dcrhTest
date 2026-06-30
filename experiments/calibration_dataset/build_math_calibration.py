from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ID_FIELDS = ["id", "task_id", "problem_id", "question_id", "unique_id"]
DEFAULT_QUESTION_FIELDS = ["problem", "question", "prompt", "Question"]
DEFAULT_ANSWER_FIELDS = [
    "answer",
    "final_answer",
    "target",
    "solution",
    "Answer",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(value)
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        for key in ["data", "train", "test", "examples", "rows"]:
            if isinstance(value.get(key), list):
                rows = value[key]
                break
        else:
            rows = [value]
    else:
        raise ValueError(f"Unsupported JSON shape: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSON rows must be objects: {path}")
    return list(rows)


def _read_delimited(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter=delimiter)]


def _read_rows(path: Path, data_format: str = "auto") -> list[dict[str, Any]]:
    fmt = data_format
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            fmt = "jsonl"
        elif suffix == ".json":
            fmt = "json"
        elif suffix == ".csv":
            fmt = "csv"
        elif suffix == ".tsv":
            fmt = "tsv"
        else:
            raise ValueError(f"Cannot infer format for {path}; set format explicitly")
    if fmt == "jsonl":
        return _read_jsonl(path)
    if fmt == "json":
        return _read_json(path)
    if fmt == "csv":
        return _read_delimited(path, ",")
    if fmt == "tsv":
        return _read_delimited(path, "\t")
    raise ValueError(f"Unsupported format {fmt!r}: {path}")


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _matches_filters(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    for field, expected in filters.items():
        actual = row.get(field)
        if isinstance(expected, list):
            if actual not in expected and str(actual) not in {str(x) for x in expected}:
                return False
        elif isinstance(expected, dict):
            if "contains" in expected:
                needle = str(expected["contains"]).lower()
                if needle not in str(actual or "").lower():
                    return False
            elif "not_in" in expected:
                blocked = {str(x) for x in expected["not_in"]}
                if str(actual) in blocked:
                    return False
            else:
                raise ValueError(f"Unsupported filter operator for {field}: {expected}")
        elif str(actual) != str(expected):
            return False
    return True


def _normalize_for_dedupe(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "row"


def _render_prompt(problem: str, prompt_style: str) -> str:
    if prompt_style == "raw":
        return problem
    if prompt_style != "aime":
        raise ValueError("prompt_style must be 'aime' or 'raw'")
    return (
        "Solve the following math problem and return ONLY the final answer.\n"
        "Please reason step by step, separate logical reasoning steps with two newline "
        "characters (\\n\\n), and put your final answer within \\boxed{}.\n\n"
        f"Problem: {problem}\n\n"
    )


def _canonicalize_rows(source: dict[str, Any], prompt_style: str) -> list[dict[str, Any]]:
    source_name = str(source["name"])
    source_path = Path(source["path"])
    difficulty = str(source.get("difficulty", source_name))
    rows = _read_rows(source_path, str(source.get("format", "auto")))
    filters = dict(source.get("filters", {}) or {})
    id_fields = list(source.get("id_fields", DEFAULT_ID_FIELDS))
    question_fields = list(source.get("question_fields", DEFAULT_QUESTION_FIELDS))
    answer_fields = list(source.get("answer_fields", DEFAULT_ANSWER_FIELDS))

    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if filters and not _matches_filters(row, filters):
            continue
        problem = _first_present(row, question_fields)
        if problem is None:
            continue
        raw_id = _first_present(row, id_fields)
        raw_id_text = str(raw_id) if raw_id is not None else f"{index:06d}"
        answer = _first_present(row, answer_fields)
        problem_text = str(problem).strip()
        result.append(
            {
                "id": f"{_safe_id(source_name)}_{_safe_id(raw_id_text)}",
                "question": _render_prompt(problem_text, prompt_style),
                "answer": None if answer is None else str(answer).strip(),
                "source": source_name,
                "difficulty": difficulty,
                "raw_id": raw_id_text,
                "problem": problem_text,
                "dedupe_key": _stable_hash(_normalize_for_dedupe(problem_text)),
            }
        )
    return result


def build_dataset(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(manifest.get("seed", 2026))
    rng = random.Random(seed)
    prompt_style = str(manifest.get("prompt_style", "aime"))
    deduplicate = bool(manifest.get("deduplicate", True))
    shuffle_output = bool(manifest.get("shuffle_output", True))

    all_rows: list[dict[str, Any]] = []
    source_stats: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for source in manifest.get("sources", []):
        rows = _canonicalize_rows(source, prompt_style=prompt_style)
        source_name = str(source["name"])
        before_dedupe = len(rows)
        kept: list[dict[str, Any]] = []
        for row in rows:
            if deduplicate and row["dedupe_key"] in seen:
                continue
            seen.add(row["dedupe_key"])
            kept.append(row)
        limit = source.get("limit")
        if limit is not None:
            rng.shuffle(kept)
            kept = kept[: int(limit)]
        all_rows.extend(kept)
        source_stats[source_name] = {
            "loaded_after_filters": before_dedupe,
            "kept": len(kept),
        }

    mix = manifest.get("mix")
    if mix:
        selected: list[dict[str, Any]] = []
        by_difficulty: dict[str, list[dict[str, Any]]] = {}
        for row in all_rows:
            by_difficulty.setdefault(str(row["difficulty"]), []).append(row)
        shortages: dict[str, dict[str, int]] = {}
        for difficulty, quota in mix.items():
            pool = list(by_difficulty.get(str(difficulty), []))
            rng.shuffle(pool)
            take = min(int(quota), len(pool))
            selected.extend(pool[:take])
            if take < int(quota):
                shortages[str(difficulty)] = {"requested": int(quota), "available": len(pool)}
    else:
        selected = list(all_rows)
        shortages = {}

    if shuffle_output:
        rng.shuffle(selected)

    for index, row in enumerate(selected):
        row["calibration_index"] = index
        row.pop("dedupe_key", None)

    summary = {
        "seed": seed,
        "prompt_style": prompt_style,
        "deduplicate": deduplicate,
        "sample_count": len(selected),
        "source_stats": source_stats,
        "difficulty_counts": {},
        "source_counts": {},
        "shortages": shortages,
    }
    for row in selected:
        summary["difficulty_counts"][row["difficulty"]] = (
            summary["difficulty_counts"].get(row["difficulty"], 0) + 1
        )
        summary["source_counts"][row["source"]] = (
            summary["source_counts"].get(row["source"], 0) + 1
        )
    return selected, summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mixed math tau calibration JSONL")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a YAML mapping")

    rows, summary = build_dataset(manifest)
    output = Path(args.output or manifest.get("output", "outputs/tau_calibration.jsonl"))
    summary_path = Path(args.summary or manifest.get("summary", str(output) + ".summary.json"))

    if not args.dry_run:
        _write_jsonl(output, rows)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if not args.dry_run:
        print(f"output={output}")
        print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
