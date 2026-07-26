#!/usr/bin/env bash
set -euxo pipefail

# DenoiseRL v2 random-token curriculum:
# random_prefix_len = floor(max_random_token * per_problem_rho).
# With v2_max_rho=1.0, the default 2048-token maximum is reachable.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export noise_source=random_tokens
export max_random_token=${max_random_token:-2048}
export random_noise_exclude_special=${random_noise_exclude_special:-True}
export v2_max_rho=${v2_max_rho:-1.0}
export TRAIN_FILE=${TRAIN_FILE:-"./data/MATH7500-train.parquet"}

exec "${script_dir}/grpo_denoise_qwen3-4b_v3.0.sh" "$@"
