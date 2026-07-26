"""Helpers for scaling DenoiseRL v2 random-token noise."""

from __future__ import annotations

import math


def scaled_random_token_count(max_random_token: int, rho: float) -> int:
    """Return ``floor(max_random_token * rho)`` after validating both inputs."""
    max_tokens = int(max_random_token)
    if max_tokens <= 0:
        raise ValueError(
            f"trainer.max_random_token must be > 0, got {max_random_token}."
        )

    ratio = float(rho)
    if not math.isfinite(ratio) or not (0.0 <= ratio <= 1.0):
        raise ValueError(f"random-token noise rho must be in [0, 1], got {rho}.")

    return min(max_tokens, int(math.floor(max_tokens * ratio)))
