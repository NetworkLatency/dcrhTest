# Math Tau Calibration Dataset Builder

This folder is an experiment-side utility. It builds an unlabeled or weakly
labeled math calibration JSONL for selecting MDRV `tau`.

For AIME24/25 target runs, mixing simple and hard sources is reasonable, but the
mixture should not be dominated by easy problems. A good first pass is:

```text
20% simple   Math500-style rows
40% medium   AMC2023-style rows
40% hard     OlympiadBench math subset
```

The calibration target is the SLM router-score distribution, not answer
accuracy. Answers are kept when present for auditing, but tau scoring does not
use them.

## Build

Copy and edit the manifest:

```bash
cp experiments/calibration_dataset/manifest.example.yaml \
  experiments/calibration_dataset/manifest.local.yaml
```

Then run:

```bash
python experiments/calibration_dataset/build_math_calibration.py \
  --manifest experiments/calibration_dataset/manifest.local.yaml
```

The output rows use the generic project schema:

```json
{
  "id": "math500_0001",
  "question": "... AIME-style prompt ...",
  "answer": "42",
  "source": "math500",
  "difficulty": "simple",
  "raw_id": "0001"
}
```

Use this file in `configs/qwen3_1p7b_4b_core.yaml` with `data.dataset: null`,
then run:

```bash
python experiments/tau_selection/score_tau.py \
  --config configs/qwen3_1p7b_4b_core.yaml \
  --rho 0.1 0.2 0.3
```

## Notes

- Do not include AIME24/25 evaluation questions in the calibration set.
- Keep the prompt style aligned with the target AIME prompt so paragraph
  boundaries and TPM behavior are comparable.
- If a source has multiple domains, use manifest `filters` to keep math-only,
  English-only, or competition-only subsets.
- If SLM pilot accuracy shows that a source is too easy or too hard, adjust its
  difficulty bucket or quota and rebuild.
