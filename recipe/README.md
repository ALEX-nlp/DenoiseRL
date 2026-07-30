# DenoiseRL recipes

The recipe directories correspond to different generations or controls of the method:

| Directory | Status | Purpose |
| --- | --- | --- |
| [`denoise`](./denoise) | Legacy v1 | Fixed-noise DenoiseRL. It mixes standard and noisy-prefix rollouts and uses a fixed or globally sampled prefix ratio. Kept for reproducing the older implementation. |
| [`denoise_v2`](./denoise_v2) | **Primary** | Current DenoiseRL: recovery-only rollout groups, per-problem adaptive `rho`, line-aligned prefixes, continuation-only loss, and stable-sample replacement. |
| [`correct_prefix`](./correct_prefix) | Control | Positive-prefix baseline built from verified-correct weak-model trajectories. |

`denoise_v3` was an experimental branch and has been removed. It is not part of the current method.

## Recommended v2 entry points

Standard GRPO:

```bash
bash recipe/denoise_v2/grpo_denoise_qwen3-4b_v2.0-line.sh
bash recipe/denoise_v2/grpo_denoise_qwen3-8b_v2.0-line.sh
```

DAPO-style dynamic sampling:

```bash
bash recipe/denoise_v2/grpo_denoise_dynamic_sample_line_qwen3-4b_v2.0.sh
bash recipe/denoise_v2/grpo_denoise_dynamic_sample_line_qwen3-8b_v2.0.sh
```

The v2 defaults use 16 recovery rollouts per problem, no additional clean rollout slots, `rho` in `[0, 0.5]`, target recovery accuracy `0.75`, update size `0.2`, and line-aligned truncation.

## Correct-prefix control

The correct-prefix recipe uses the same curriculum machinery but reverses the control direction:

```text
rho <- clip(rho - alpha * (accuracy - target_accuracy), min_rho, max_rho)
```

It begins with the configured maximum amount of correct assistance. Easy problems receive progressively shorter correct prefixes, while hard problems receive longer ones. Prepare verified-correct weak-model trajectories and run:

```bash
python recipe/correct_prefix/data_prepare.py \
  --model /path/to/weak-model \
  --dataset /path/to/train.parquet \
  --rollout-n 16 \
  --output-dir ./data/correct_prefix

bash recipe/correct_prefix/grpo_correct_prefix_qwen3-4b_v1.0.sh
bash recipe/correct_prefix/grpo_correct_prefix_qwen3-8b_v1.0.sh
```

See [`correct_prefix/README.md`](./correct_prefix/README.md) for details.
