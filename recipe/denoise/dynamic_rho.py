"""Feedback controllers for the DenoiseRL partial-prefix ratio ``rho``.

Two feedback signals are supported:

``recoverability`` (legacy ``dynamic`` strategy)
    ``clip(acc_noise / acc_base, 0, 1)``. This requires both base and noisy
    rollouts in a batch.

``accuracy`` (``dynamic_acc`` strategy)
    The current batch's overall verifier accuracy. This is intended for
    denoise-only runs, where there are no base rollouts to form a recoverability
    ratio.

For either signal, larger-than-target feedback increases ``rho`` (more noise),
and smaller-than-target feedback decreases it::

    rho <- clip(rho + alpha * (feedback - target), min_rho, max_rho)
"""

import math


class DynamicRhoController:
    """Stateful feedback controller for ``part_response_ratio``."""

    RECOVERABILITY = "recoverability"
    ACCURACY = "accuracy"

    def __init__(
        self,
        min_rho: float = 0.1,
        max_rho: float = 0.5,
        initial_rho: float = 0.2,
        target_recoverability: float = 0.8,
        alpha: float = 0.05,
        *,
        feedback: str = RECOVERABILITY,
        target_accuracy: float = 0.75,
    ) -> None:
        self.feedback = str(feedback).strip().lower()
        if self.feedback not in (self.RECOVERABILITY, self.ACCURACY):
            raise ValueError(
                "dynamic rho feedback must be 'recoverability' or 'accuracy', "
                f"got {feedback!r}."
            )

        self.min_rho = self._finite_float("dynamic_rho_min", min_rho)
        self.max_rho = self._finite_float("dynamic_rho_max", max_rho)
        if not (0.0 <= self.min_rho <= self.max_rho <= 1.0):
            raise ValueError(
                "trainer.dynamic_rho_min/max must satisfy "
                f"0 <= min <= max <= 1, got min={self.min_rho}, max={self.max_rho}."
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

        self.target_accuracy = self._finite_float(
            "dynamic_rho_target_accuracy", target_accuracy
        )
        if not (0.0 <= self.target_accuracy <= 1.0):
            raise ValueError(
                "trainer.dynamic_rho_target_accuracy must be in [0, 1], "
                f"got {self.target_accuracy}."
            )

        self.alpha = self._finite_float("dynamic_rho_alpha", alpha)
        if self.alpha < 0.0:
            raise ValueError(f"trainer.dynamic_rho_alpha must be >= 0, got {self.alpha}.")

        self.num_updates = 0
        self.last_acc_base = None
        self.last_acc_noise = None
        self.last_recoverability = None
        self.last_batch_accuracy = None
        self.last_error = None
        self.last_delta = 0.0

    @classmethod
    def from_trainer_config(
        cls, cfg, *, feedback: str = RECOVERABILITY
    ) -> "DynamicRhoController":
        """Create a controller from ``config.trainer`` with stable defaults."""
        accuracy_feedback = feedback == cls.ACCURACY
        return cls(
            min_rho=cfg.get("dynamic_rho_min", 0.0 if accuracy_feedback else 0.1),
            max_rho=cfg.get("dynamic_rho_max", 0.5),
            initial_rho=cfg.get("dynamic_rho_initial", 0.0 if accuracy_feedback else 0.2),
            target_recoverability=cfg.get("dynamic_rho_target_recoverability", 0.8),
            alpha=cfg.get("dynamic_rho_alpha", 0.05),
            feedback=feedback,
            target_accuracy=cfg.get("dynamic_rho_target_accuracy", 0.75),
        )

    @staticmethod
    def _finite_float(name: str, value) -> float:
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"trainer.{name} must be finite, got {value!r}.")
        return out

    @staticmethod
    def _validate_accuracy(name: str, value) -> float:
        out = DynamicRhoController._finite_float(name, value)
        if not (0.0 <= out <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {out}.")
        return out

    def sample(self) -> float:
        """Return the rho value to use for the next noisy prefix."""
        return self.current_rho

    def _apply_feedback(self, feedback_value: float, target: float) -> tuple[float, float, float]:
        error = feedback_value - target
        old_rho = self.current_rho
        new_rho = min(self.max_rho, max(self.min_rho, old_rho + self.alpha * error))

        self.current_rho = new_rho
        self.num_updates += 1
        self.last_error = error
        self.last_delta = new_rho - old_rho
        return old_rho, new_rho, error

    def _skipped_metrics(self, reason: str) -> dict:
        metrics = self.metrics()
        metrics.update(
            {
                "denoise/dynamic_rho/update_applied": 0.0,
                "denoise/dynamic_rho/update_skipped_missing_acc": float(reason == "missing_acc"),
                "denoise/dynamic_rho/update_skipped_zero_base": float(reason == "zero_base"),
            }
        )
        return metrics

    def update_from_acc(self, acc_base, acc_noise) -> dict:
        """Update from base/noise accuracy (legacy recoverability controller)."""
        if acc_base is None or acc_noise is None:
            return self._skipped_metrics("missing_acc")

        acc_base_f = self._validate_accuracy("reward_model/acc_base", acc_base)
        acc_noise_f = self._validate_accuracy("reward_model/acc_noise", acc_noise)
        if acc_base_f == 0.0:
            metrics = self._skipped_metrics("zero_base")
            metrics.update(
                {
                    "denoise/dynamic_rho/acc_base": acc_base_f,
                    "denoise/dynamic_rho/acc_noise": acc_noise_f,
                }
            )
            return metrics

        recoverability = min(1.0, max(0.0, acc_noise_f / acc_base_f))
        old_rho, new_rho, error = self._apply_feedback(
            recoverability, self.target_recoverability
        )

        self.last_acc_base = acc_base_f
        self.last_acc_noise = acc_noise_f
        self.last_recoverability = recoverability

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

    def update_from_accuracy(self, batch_accuracy) -> dict:
        """Update rho directly from the current batch's overall accuracy."""
        if batch_accuracy is None:
            return self._skipped_metrics("missing_acc")

        batch_accuracy_f = self._validate_accuracy("reward_model/acc", batch_accuracy)
        old_rho, new_rho, error = self._apply_feedback(
            batch_accuracy_f, self.target_accuracy
        )
        self.last_batch_accuracy = batch_accuracy_f

        metrics = self.metrics()
        metrics.update(
            {
                "denoise/dynamic_rho/update_applied": 1.0,
                "denoise/dynamic_rho/update_skipped_missing_acc": 0.0,
                "denoise/dynamic_rho/update_skipped_zero_base": 0.0,
                "denoise/dynamic_rho/batch_accuracy": batch_accuracy_f,
                "denoise/dynamic_rho/accuracy_error": error,
                "denoise/dynamic_rho/rho_before_update": old_rho,
                "denoise/dynamic_rho/rho_after_update": new_rho,
                "denoise/dynamic_rho/rho_update_delta": self.last_delta,
            }
        )
        return metrics

    def update_from_metrics(self, metrics: dict) -> dict:
        """Update rho using metrics emitted by the trainer."""
        if self.feedback == self.ACCURACY:
            return self.update_from_accuracy(metrics.get("reward_model/acc"))
        return self.update_from_acc(
            metrics.get("reward_model/acc_base"),
            metrics.get("reward_model/acc_noise"),
        )

    def metrics(self) -> dict:
        """Return controller state for logging."""
        out = {
            "denoise/dynamic_rho/enabled": 1.0,
            "denoise/dynamic_rho/feedback_is_accuracy": float(
                self.feedback == self.ACCURACY
            ),
            "denoise/dynamic_rho/current_rho": self.current_rho,
            "denoise/dynamic_rho/min_rho": self.min_rho,
            "denoise/dynamic_rho/max_rho": self.max_rho,
            "denoise/dynamic_rho/alpha": self.alpha,
            "denoise/dynamic_rho/num_updates": float(self.num_updates),
        }
        if self.feedback == self.ACCURACY:
            out["denoise/dynamic_rho/target_accuracy"] = self.target_accuracy
            if self.last_batch_accuracy is not None:
                out["denoise/dynamic_rho/last_batch_accuracy"] = self.last_batch_accuracy
                out["denoise/dynamic_rho/last_accuracy_error"] = self.last_error
                out["denoise/dynamic_rho/last_rho_update_delta"] = self.last_delta
        else:
            out["denoise/dynamic_rho/target_recoverability"] = self.target_recoverability
            if self.last_recoverability is not None:
                out["denoise/dynamic_rho/last_recoverability"] = self.last_recoverability
                out["denoise/dynamic_rho/last_recoverability_error"] = self.last_error
                out["denoise/dynamic_rho/last_rho_update_delta"] = self.last_delta
        return out
