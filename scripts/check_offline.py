from __future__ import annotations

import argparse
from pathlib import Path

import torch
import transformers
from packaging.version import Version
from transformers import AutoConfig, AutoTokenizer

from dcrh.config import load_config
from dcrh.utils.offline import force_offline_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local paths and pinned runtime without loading weights")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    force_offline_environment()
    cfg = load_config(args.config)
    cfg.validate(require_references=False)
    if Version(transformers.__version__) != Version("4.57.6"):
        raise RuntimeError(
            f"This release requires transformers==4.57.6; found {transformers.__version__}"
        )
    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")
    for role, model in (("slm", cfg.models.slm), ("llm", cfg.models.llm)):
        path = Path(model.path)
        if not path.exists():
            raise FileNotFoundError(path)
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=model.trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=model.trust_remote_code)
        print(
            f"{role}: path={path.resolve()} model_type={config.model_type} "
            f"layers={config.num_hidden_layers} vocab={len(tokenizer)} device={model.device}"
        )
    print("offline check passed")


if __name__ == "__main__":
    main()
