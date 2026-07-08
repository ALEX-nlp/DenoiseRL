#!/usr/bin/env bash
set -euxo pipefail

# Pure DAPO baseline: no noisy sub-rollouts. Keep total rollouts per problem at 16.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export n_resp_per_prompt=${n_resp_per_prompt:-16}
export sub_rollout_k=${sub_rollout_k:-0}
export TRAIN_FILE=${TRAIN_FILE:-"./data/MATH7500-train.parquet"}
export wandb_run_id=${wandb_run_id:-"BASELINE_DAPO_v1.0_Qwen3-1.7B-Base_n${n_resp_per_prompt}"}
export exp_name=${exp_name:-"baseline-dapo-v1.0-model-Qwen3-1.7B-Base-lr-1e-6-bsz-16-n_resp-${n_resp_per_prompt}-mini-16"}

exec bash "${SCRIPT_DIR}/dapo_denoise_qwen3-1.7b_v1.0.sh" "$@"
