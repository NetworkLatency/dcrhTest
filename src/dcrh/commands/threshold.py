from __future__ import annotations

import argparse

import numpy as np

from ..utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-select B from saved budget maxima")
    parser.add_argument("--threshold-json", required=True)
    parser.add_argument("--target-alarm-rate", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 < args.target_alarm_rate < 1.0:
        raise ValueError("target-alarm-rate must be in (0, 1)")
    payload = read_json(args.threshold_json)
    maxima = np.asarray(payload["budget_max_scores"], dtype=np.float64)
    q = 1.0 - args.target_alarm_rate
    threshold = max(1e-12, float(np.quantile(maxima, q, method="higher")))
    output = dict(payload)
    output["threshold"] = threshold
    output["target_problem_alarm_rate"] = args.target_alarm_rate
    output["selection_quantile"] = q
    write_json(args.output, output)
    print(f"threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
