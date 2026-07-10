"""Feedback controller for dynamic DenoiseRL prefix ratio.

The controller keeps one scalar ``rho`` and updates it from the observed
recoverability of noisy rollouts:

    recoverability = clip(acc_noise / acc_base, 0, 1)
    rho <- clip(rho + alpha * (recoverability - target), min_rho, max_rho)

When noisy rollouts retain less of the base accuracy than the target, ``rho``
decreases. When they retain more, ``rho`` increases. If base accuracy is zero,
recoverability is undefined and the update is skipped.
"""

import math


class DynamicRhoController:
    """Stateful feedback controller for ``part_response_ratio``."""

    def __init__(
        self,
        min_rho: float = 0.1,
        max_rho: float = 0.5,
        initial_rho: float = 0.2,
        target_recoverability: float = 0.8,
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

        self.target_recoverability = self._finite_float(
            "dynamic_rho_target_recoverability", target_recoverability
        )
        if not (0.0 < self.target_recoverability <= 1.0):
            raise ValueError(
                "trainer.dynamic_rho_target_recoverability must be in (0, 1], "
                f"got {self.target_recoverability}."
            )

        self.alpha = self._finite_float("dynamic_rho_alpha", alpha)
        if self.alpha < 0.0:
            raise ValueError(f"trainer.dynamic_rho_alpha must be >= 0, got {self.alpha}.")

        self.num_updates = 0
        self.last_acc_base = None
        self.last_acc_noise = None
        self.last_recoverability = None
        self.last_error = None
        self.last_delta = 0.0

    @classmethod
    def from_trainer_config(cls, cfg) -> "DynamicRhoController":
        """Create a controller from ``config.trainer`` with stable defaults."""
        return cls(
            min_rho=cfg.get("dynamic_rho_min", 0.1),
            max_rho=cfg.get("dynamic_rho_max", 0.5),
            initial_rho=cfg.get("dynamic_rho_initial", 0.2),
            target_recoverability=cfg.get("dynamic_rho_target_recoverability", 0.8),
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
            metrics["denoise/dynamic_rho/update_skipped_zero_base"] = 0.0
            return metrics

        acc_base_f = self._finite_float("reward_model/acc_base", acc_base)
        acc_noise_f = self._finite_float("reward_model/acc_noise", acc_noise)
        if not (0.0 <= acc_base_f <= 1.0):
            raise ValueError(f"reward_model/acc_base must be in [0, 1], got {acc_base_f}.")
        if not (0.0 <= acc_noise_f <= 1.0):
            raise ValueError(f"reward_model/acc_noise must be in [0, 1], got {acc_noise_f}.")

        if acc_base_f == 0.0:
            metrics = self.metrics()
            metrics.update(
                {
                    "denoise/dynamic_rho/update_applied": 0.0,
                    "denoise/dynamic_rho/update_skipped_missing_acc": 0.0,
                    "denoise/dynamic_rho/update_skipped_zero_base": 1.0,
                    "denoise/dynamic_rho/acc_base": acc_base_f,
                    "denoise/dynamic_rho/acc_noise": acc_noise_f,
                }
            )
            return metrics

        recoverability = min(1.0, max(0.0, acc_noise_f / acc_base_f))
        error = recoverability - self.target_recoverability
        old_rho = self.current_rho
        delta = self.alpha * error
        new_rho = min(self.max_rho, max(self.min_rho, old_rho + delta))

        self.current_rho = new_rho
        self.num_updates += 1
        self.last_acc_base = acc_base_f
        self.last_acc_noise = acc_noise_f
        self.last_recoverability = recoverability
        self.last_error = error
        self.last_delta = new_rho - old_rho

        metrics = self.metrics()
        metrics.update(
            {
                "denoise/dynamic_rho/update_applied": 1.0,
                "denoise/dynamic_rho/update_skipped_missing_acc": 0.0,
                "denoise/dynamic_rho/update_skipped_zero_base": 0.0,
                "denoise/dynamic_rho/acc_base": acc_base_f,
                "denoise/dynamic_rho/acc_noise": acc_noise_f,
                "denoise/dynamic_rho/recoverability": recoverability,
                "denoise/dynamic_rho/recoverability_error": error,
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
            "denoise/dynamic_rho/target_recoverability": self.target_recoverability,
            "denoise/dynamic_rho/alpha": self.alpha,
            "denoise/dynamic_rho/num_updates": float(self.num_updates),
        }
        if self.last_recoverability is not None:
            out["denoise/dynamic_rho/last_recoverability"] = self.last_recoverability
            out["denoise/dynamic_rho/last_recoverability_error"] = self.last_error
            out["denoise/dynamic_rho/last_rho_update_delta"] = self.last_delta
        return out
