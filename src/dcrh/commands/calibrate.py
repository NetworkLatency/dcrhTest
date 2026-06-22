from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..core.calibration import calibrate_model
from ..utils.offline import force_offline_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one model's unlabeled DCRH reference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", choices=["slm", "llm"], required=True)
    parser.add_argument("--reference-out", required=True)
    parser.add_argument("--threshold-out", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    force_offline_environment()
    cfg = load_config(args.config)
    report = calibrate_model(
        cfg=cfg,
        role=args.role,
        output_reference=Path(args.reference_out),
        output_threshold=Path(args.threshold_out),
    )
    print(f"reference={report['reference_path']}")
    if report["threshold"] is None:
        print("threshold=not-used-for-llm-reference")
    else:
        print(f"threshold={report['threshold']:.6f}")
    print(f"token_mass={report['token_mass']}")


if __name__ == "__main__":
    main()
