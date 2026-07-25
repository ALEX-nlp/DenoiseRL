# Correct-prefix control recipe

This directory is an independent copy of the DenoiseRL v2 training path,
adapted into a positive-prefix control. The original `recipe/denoise_v2`
directory is not imported or modified.

First collect verified-correct weak-model rollouts:

```bash
python recipe/correct_prefix/data_prepare.py \
  --model /path/to/weak-model \
  --dataset ./data/MATH7500.parquet \
  --rollout-n 16 \
  --output-dir ./data/correct_prefix
```

The generated parquet contains `correct_answer_with_boxed: list[str]`. Set
`TRAIN_FILE` to that parquet, then run one of:

```bash
bash recipe/correct_prefix/grpo_correct_prefix_qwen3-4b_v1.0.sh
bash recipe/correct_prefix/grpo_correct_prefix_qwen3-8b_v1.0.sh
```

The per-problem control law is the reverse of denoise v2:

```text
rho <- clip(rho - alpha * (accuracy - target_accuracy), min_rho, max_rho)
```

Both scripts start at their configured maximum `rho`: successful problems
receive progressively shorter correct prefixes, while struggling problems
receive longer prefixes.
