#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fixed rho=0.2, rounded to the nearest complete line boundary. Wrong solutions
# without an interior newline fall back to the legacy token cut.
export part_response_ratio_strategy=fixed
export part_response_ratio_fixed=${part_response_ratio_fixed:-0.2}
export partial_wrong_cut_strategy=line

exec bash "${SCRIPT_DIR}/denoise_qwen3-4b_v1.0.sh" "$@"
