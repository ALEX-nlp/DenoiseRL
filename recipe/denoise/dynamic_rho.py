"""Feedback controller for dynamic DenoiseRL prefix ratio.

The controller keeps one scalar ``rho`` and updates it from the observed
``acc_base - acc_noise`` gap:

    rho <- clip(rho - alpha * (gap - target_gap), min_rho, max_rho)

When noisy rollouts are too hard, the gap is large and ``rho`` decreases. When
the gap is smaller than the target, ``rho`` increases.
"""

import math


class DynamicRhoController:
    """Stateful feedback controller for ``part_response_ratio``."""

    def __init__(
        self,
        min_rho: float = 0.1,
        max_rho: float = 0.5,
        initial_rho: float = 0.2,
        target_gap: float = 0.1,
        alpha: float = 0.05,
    ) -> None:
        self.min_rho = self._finite_float("dynamic_rho_min", min_rho)
        self.max_rho = self._finite_float("dynamic_rho_max", max_rho)
        if not (0.0 < self.min_rho <= self.max_rho <= 1.0):
            raise ValueError(
                "trainer.dynamic_rho_min/max must satisfy "
                f"0 < min <= max <= 1, got min={self.min_rho}, max={self.max_rho}."
            )

        initial = self._finite_float("dynamic_rho_initial", initial_rho)
        if not (self.min_rho <= initial <= self.max_rho):
            raise ValueError(
                "trainer.dynamic_rho_initial must be within "
                f"[{self.min_rho}, {self.max_rho}], got {initial}."
            )
        self.current_rho = initial

        self.target_gap = self._finite_float("dynamic_rho_target_gap", target_gap)
        if self.target_gap < 0.0:
            raise ValueError(
                f"trainer.dynamic_rho_target_gap must be >= 0, got {self.target_gap}."
            )

        self.alpha = self._finite_float("dynamic_rho_alpha", alpha)
        if self.alpha < 0.0:
            raise ValueError(f"trainer.dynamic_rho_alpha must be >= 0, got {self.alpha}.")

        self.num_updates = 0
        self.last_acc_base = None
        self.last_acc_noise = None
        self.last_gap = None
        self.last_error = None
        self.last_delta = 0.0

    @classmethod
    def from_trainer_config(cls, cfg) -> "DynamicRhoController":
        """Create a controller from ``config.trainer`` with stable defaults."""
        return cls(
            min_rho=cfg.get("dynamic_rho_min", 0.1),
            max_rho=cfg.get("dynamic_rho_max", 0.5),
            initial_rho=cfg.get("dynamic_rho_initial", 0.2),
            target_gap=cfg.get(
                "dynamic_rho_target_gap", cfg.get("dynamic_rho_target_diff", 0.1)
            ),
            alpha=cfg.get("dynamic_rho_alpha", 0.05),
        )

    @staticmethod
    def _finite_float(name: str, value) -> float:
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"trainer.{name} must be finite, got {value!r}.")
        return out

    def sample(self) -> float:
        """Return the rho value to use for the next noisy prefix."""
        return self.current_rho

    def update_from_acc(self, acc_base, acc_noise) -> dict:
        """Update rho from observed base/noise accuracies and return metrics."""
        if acc_base is None or acc_noise is None:
            metrics = self.metrics()
            metrics["denoise/dynamic_rho/update_applied"] = 0.0
            metrics["denoise/dynamic_rho/update_skipped_missing_acc"] = 1.0
            return metrics

        acc_base_f = self._finite_float("reward_model/acc_base", acc_base)
        acc_noise_f = self._finite_float("reward_model/acc_noise", acc_noise)
        gap = acc_base_f - acc_noise_f
        error = gap - self.target_gap
        old_rho = self.current_rho
        delta = -self.alpha * error
        new_rho = min(self.max_rho, max(self.min_rho, old_rho + delta))

        self.current_rho = new_rho
        self.num_updates += 1
        self.last_acc_base = acc_base_f
        self.last_acc_noise = acc_noise_f
        self.last_gap = gap
        self.last_error = error
        self.last_delta = new_rho - old_rho

        metrics = self.metrics()
        metrics.update(
            {
                "denoise/dynamic_rho/update_applied": 1.0,
                "denoise/dynamic_rho/update_skipped_missing_acc": 0.0,
                "denoise/dynamic_rho/acc_base": acc_base_f,
                "denoise/dynamic_rho/acc_noise": acc_noise_f,
                "denoise/dynamic_rho/acc_gap_base_minus_noise": gap,
                "denoise/dynamic_rho/gap_error": error,
                "denoise/dynamic_rho/rho_before_update": old_rho,
                "denoise/dynamic_rho/rho_after_update": new_rho,
                "denoise/dynamic_rho/rho_update_delta": self.last_delta,
            }
        )
        return metrics

    def update_from_metrics(self, metrics: dict) -> dict:
        """Update rho using metrics emitted by the trainer."""
        return self.update_from_acc(
            metrics.get("reward_model/acc_base"),
            metrics.get("reward_model/acc_noise"),
        )

    def metrics(self) -> dict:
        """Return controller state for logging."""
        out = {
            "denoise/dynamic_rho/enabled": 1.0,
            "denoise/dynamic_rho/current_rho": self.current_rho,
            "denoise/dynamic_rho/min_rho": self.min_rho,
            "denoise/dynamic_rho/max_rho": self.max_rho,
            "denoise/dynamic_rho/target_gap": self.target_gap,
            "denoise/dynamic_rho/alpha": self.alpha,
            "denoise/dynamic_rho/num_updates": float(self.num_updates),
        }
        if self.last_gap is not None:
            out["denoise/dynamic_rho/last_acc_gap_base_minus_noise"] = self.last_gap
            out["denoise/dynamic_rho/last_gap_error"] = self.last_error
            out["denoise/dynamic_rho/last_rho_update_delta"] = self.last_delta
        return out
