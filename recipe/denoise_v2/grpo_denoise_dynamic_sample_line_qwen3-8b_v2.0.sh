#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export model_name=${model_name:-Qwen3-8B-Base}
export project_name=${project_name:-DenoiseRL-v2-8B}

exec "${script_dir}/grpo_denoise_dynamic_sample_line_qwen3-4b_v2.0.sh" "$@"
