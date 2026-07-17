#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Denoise-only control run with a constant partial-wrong prefix ratio rho=0.2.
export part_response_ratio_strategy=fixed
export part_response_ratio_fixed=${part_response_ratio_fixed:-0.2}

exec bash "${SCRIPT_DIR}/denoise_only_qwen3-4b_v1.0.sh" "$@"
