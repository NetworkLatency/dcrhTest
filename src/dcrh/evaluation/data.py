from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..config import DataConfig
from .benchmarks import build_benchmark_example


@dataclass(slots=True)
class Example:
    example_id: str
    question: str
    answer: str | None
    raw: dict[str, Any]


def _coerce_id(value: Any, index: int) -> str:
    if value is None:
        return str(index)
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_no} must contain a JSON object per line"
                )
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
        raise ValueError(f"Unsupported JSON dataset shape in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSON dataset rows must be objects: {path}")
    return list(rows)


def _read_delimited(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter=delimiter)]


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas and pyarrow are required for data.format=parquet"
        ) from exc
    return pd.read_parquet(path).to_dict(orient="records")


def _resolve_format(path: Path, configured: str) -> str:
    if configured != "auto":
        return configured
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix == ".parquet":
        return "parquet"
    if path.is_dir():
        return "hf_disk"
    raise ValueError(
        f"Cannot infer data.format from {path}; set data.format explicitly"
    )


def load_local_rows(path: str | Path, data_format: str = "auto") -> list[dict[str, Any]]:
    source = Path(path)
    fmt = _resolve_format(source, data_format)
    if fmt == "jsonl":
        return _read_jsonl(source)
    if fmt == "json":
        return _read_json(source)
    if fmt == "csv":
        return _read_delimited(source, ",")
    if fmt == "tsv":
        return _read_delimited(source, "\t")
    if fmt == "parquet":
        return _read_parquet(source)
    raise ValueError(f"Unsupported local data format {fmt!r}: {source}")


def _example_from_row(row: dict[str, Any], index: int, cfg: DataConfig) -> Example:
    configured_id = row.get(cfg.id_field) if cfg.id_field else None
    if cfg.dataset is not None:
        bench = build_benchmark_example(
            row=row,
            dataset=cfg.dataset,
            index=index,
            configured_id=None if configured_id is None else str(configured_id),
        )
        return Example(
            example_id=bench.example_id,
            question=bench.question,
            answer=bench.answer,
            raw=row,
        )

    if cfg.question_field not in row:
        raise KeyError(
            f"Missing question field {cfg.question_field!r} at dataset row {index}"
        )
    answer = None
    if cfg.answer_field is not None and cfg.answer_field in row:
        value = row[cfg.answer_field]
        answer = None if value is None else str(value)
    return Example(
        example_id=_coerce_id(configured_id, index),
        question=str(row[cfg.question_field]),
        answer=answer,
        raw=row,
    )


def _iter_local_rows(cfg: DataConfig) -> Iterator[Example]:
    rows = load_local_rows(cfg.path, cfg.format)
    stop = len(rows) if cfg.limit is None else min(len(rows), cfg.start + cfg.limit)
    for index in range(cfg.start, stop):
        yield _example_from_row(rows[index], index, cfg)


def _iter_hf_disk(cfg: DataConfig) -> Iterator[Example]:
    try:
        from datasets import Dataset, DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for data.format=hf_disk. Install the optional dependency locally."
        ) from exc

    obj = load_from_disk(cfg.path)
    if isinstance(obj, DatasetDict):
        if not cfg.split:
            raise ValueError("data.split is required when loading a DatasetDict from disk")
        dataset = obj[cfg.split]
    elif isinstance(obj, Dataset):
        dataset = obj
    else:
        raise TypeError(f"Unsupported object returned by load_from_disk: {type(obj)!r}")

    stop = len(dataset) if cfg.limit is None else min(len(dataset), cfg.start + cfg.limit)
    for index in range(cfg.start, stop):
        row = dict(dataset[index])
        yield _example_from_row(row, index, cfg)


def iter_examples(cfg: DataConfig) -> Iterator[Example]:
    data_format = _resolve_format(Path(cfg.path), cfg.format)
    if data_format == "hf_disk":
        yield from _iter_hf_disk(cfg)
    else:
        yield from _iter_local_rows(cfg)
