#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Feedback-controlled partial-wrong prefix ratio for Qwen3-4B GRPO.
# Target: noisy rollouts retain 80% of base-rollout accuracy.
export part_response_ratio_strategy=${part_response_ratio_strategy:-dynamic}
export dynamic_rho_min=${dynamic_rho_min:-0.1}
export dynamic_rho_max=${dynamic_rho_max:-0.5}
export dynamic_rho_initial=${dynamic_rho_initial:-0.2}
export dynamic_rho_target_recoverability=${dynamic_rho_target_recoverability:-0.8}
export dynamic_rho_alpha=${dynamic_rho_alpha:-0.05}

exec bash "${SCRIPT_DIR}/denoise_qwen3-4b_v1.0.sh" "$@"
