#!/usr/bin/env bash
set -euxo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# A positive value is subtracted from every rollout that reaches
# max_response_length. Set it to 0 to disable the branch.
export response_clip_reward_penalty=${response_clip_reward_penalty:-0}
export correct_length_reward_enabled=False

exec "${script_dir}/grpo_denoise_qwen3-4b_v3.0.sh"
