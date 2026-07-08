#!/usr/bin/env bash
set -euxo pipefail

# GRPO entrypoint for random token noise. Override random_noise_len to choose
# how many tokenizer ids are prepended before each sub-rollout response.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export noise_source=random_tokens
export random_noise_len=${random_noise_len:-512}
export random_noise_exclude_special=${random_noise_exclude_special:-True}
export TRAIN_FILE=${TRAIN_FILE:-"./data/MATH7500-train.parquet"}

exec bash "${SCRIPT_DIR}/denoise_qwen3-1.7b_v1.0.sh" "$@"
