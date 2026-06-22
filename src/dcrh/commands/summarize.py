from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DCRH JSONL results")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    with Path(args.results).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("No result rows found")

    terminal = Counter(row["terminal_reason"] for row in rows)
    correct_values = [
        row.get("evaluation", {}).get("correct")
        for row in rows
        if row.get("evaluation", {}).get("correct") is not None
    ]
    sums = defaultdict(float)
    for row in rows:
        counters = row["cost"]["counters"]
        for key, value in counters.items():
            if isinstance(value, (int, float)):
                sums[key] += value
    n = len(rows)
    summary = {
        "examples": n,
        "accuracy": (
            sum(bool(x) for x in correct_values) / len(correct_values)
            if correct_values
            else None
        ),
        "terminal_reasons": dict(terminal),
        "average_cost_counters": {k: v / n for k, v in sorted(sums.items())},
        "total_cost_counters": dict(sorted(sums.items())),
    }
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
