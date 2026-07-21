"""Ordered-pool, per-sample noise curriculum for DenoiseRL v2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


def _normalize_id(value):
    return value.item() if isinstance(value, np.generic) else value


def has_usable_wrong_solution(wrong_solutions) -> bool:
    """Return whether a pool row can produce a DenoiseRL noisy prefix."""
    if wrong_solutions is None:
        return False
    if isinstance(wrong_solutions, str):
        candidates = [wrong_solutions]
    else:
        try:
            candidates = list(wrong_solutions)
        except TypeError:
            return False
    return any(
        isinstance(candidate, str) and bool(candidate.strip())
        for candidate in candidates
    )


@dataclass
class SampleNoiseState:
    rho: float
    # Rhos actually used for rollout, newest last. Only the configured window is
    # retained; num_samples separately records the full sampling count.
    history: list[float] = field(default_factory=list)
    num_samples: int = 0


class PerSampleNoiseCurriculum:
    """Maintain one fixed noise ratio per problem in a deterministic active batch."""

    STATE_VERSION = 2

    def __init__(
        self,
        problem_ids: Sequence,
        batch_size: int,
        *,
        initial_rho: float = 0.0,
        min_rho: float = 0.0,
        max_rho: float = 0.5,
        target_accuracy: float = 0.75,
        alpha: float = 0.2,
        history_window: int = 10,
        min_history: int = 2,
        slope_threshold: float = 0.0075,
    ) -> None:
        self.problem_ids = tuple(_normalize_id(pid) for pid in problem_ids)
        if not self.problem_ids:
            raise ValueError("DenoiseRL v2 requires a non-empty training pool.")
        try:
            unique_problem_ids = set(self.problem_ids)
        except TypeError as exc:
            raise ValueError("Every problem_id must be hashable.") from exc
        if len(unique_problem_ids) != len(self.problem_ids):
            raise ValueError("DenoiseRL v2 requires a unique problem_id per pool row.")
        self.problem_id_to_index = {
            problem_id: index for index, problem_id in enumerate(self.problem_ids)
        }

        self.batch_size = int(batch_size)
        if not (1 <= self.batch_size <= len(self.problem_ids)):
            raise ValueError(
                "batch_size must be in [1, pool_size], got "
                f"{self.batch_size} for pool_size={len(self.problem_ids)}."
            )

        self.min_rho = self._finite("min_rho", min_rho)
        self.max_rho = self._finite("max_rho", max_rho)
        self.initial_rho = self._finite("initial_rho", initial_rho)
        if not (0.0 <= self.min_rho <= self.max_rho <= 1.0):
            raise ValueError("rho bounds must satisfy 0 <= min_rho <= max_rho <= 1.")
        if not (self.min_rho <= self.initial_rho <= self.max_rho):
            raise ValueError("initial_rho must lie within the configured rho bounds.")

        self.target_accuracy = self._finite("target_accuracy", target_accuracy)
        if not (0.0 <= self.target_accuracy <= 1.0):
            raise ValueError("target_accuracy must be in [0, 1].")
        self.alpha = self._finite("alpha", alpha)
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative.")

        self.history_window = int(history_window)
        self.min_history = int(min_history)
        if self.history_window < 2:
            raise ValueError("history_window must be at least 2.")
        if not (2 <= self.min_history <= self.history_window):
            raise ValueError("min_history must be in [2, history_window].")
        self.slope_threshold = self._finite("slope_threshold", slope_threshold)
        if self.slope_threshold < 0.0:
            raise ValueError("slope_threshold must be non-negative.")

        # Pool order is never shuffled: the first batch is active, and the cursor
        # always advances through one consecutive slice for replacements.
        self.active_indices = list(range(self.batch_size))
        self.next_pool_index = self.batch_size
        self.states = {
            index: SampleNoiseState(rho=self.initial_rho)
            for index in self.active_indices
        }
        self.retired_total = 0
        self.pool_round = 1
        self.rounds_completed = 0
        self.samples_introduced_total = self.batch_size

    @staticmethod
    def _finite(name: str, value) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite, got {value!r}.")
        return result

    @property
    def pool_size(self) -> int:
        return len(self.problem_ids)

    @property
    def active_problem_ids(self) -> tuple:
        return tuple(self.problem_ids[index] for index in self.active_indices)

    def rho_for_problem(self, problem_id) -> float:
        normalized = _normalize_id(problem_id)
        index = self.problem_id_to_index.get(normalized)
        if index is None or index not in self.states:
            raise KeyError(f"problem_id {normalized!r} is not in the active v2 batch.")
        return self.states[index].rho

    def mean_rho(self) -> float:
        return float(np.mean([self.states[index].rho for index in self.active_indices]))

    def _slope(self, state: SampleNoiseState) -> float | None:
        # This is exactly the most recent min(actual sample count, window) rhos.
        # With fewer than two actual samples a linear slope is undefined, so the
        # sample cannot be retired yet.
        values = state.history[-self.history_window :]
        if len(values) < self.min_history:
            return None
        x = np.arange(len(values), dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)
        x -= np.mean(x)
        denominator = float(np.dot(x, x))
        return float(np.dot(x, y - np.mean(y)) / denominator)

    def update(self, problem_id_to_average_accuracy: Mapping) -> dict[str, float]:
        """Update every rho independently, then replace stable samples in order."""
        accuracies = {
            _normalize_id(problem_id): accuracy
            for problem_id, accuracy in problem_id_to_average_accuracy.items()
        }
        used_rhos: list[float] = []
        average_accuracies: list[float] = []
        for index in self.active_indices:
            problem_id = self.problem_ids[index]
            if problem_id not in accuracies:
                raise ValueError(
                    f"Missing rollout average accuracy for active problem_id {problem_id!r}."
                )
            accuracy = self._finite(
                f"average accuracy for problem_id {problem_id!r}",
                accuracies[problem_id],
            )
            if not (0.0 <= accuracy <= 1.0):
                raise ValueError(
                    f"Average accuracy for problem_id {problem_id!r} must be in [0, 1]."
                )

            state = self.states[index]
            used_rho = state.rho
            state.history.append(used_rho)
            state.history = state.history[-self.history_window :]
            state.num_samples += 1
            # rho=0 rows use clean/base accuracy; rho>0 rows use noisy accuracy.
            # In both cases the 16 rollout rewards are averaged per problem before
            # reaching this controller.
            state.rho = min(
                self.max_rho,
                max(
                    self.min_rho,
                    used_rho + self.alpha * (accuracy - self.target_accuracy),
                ),
            )
            used_rhos.append(used_rho)
            average_accuracies.append(accuracy)

        slopes = {
            index: self._slope(self.states[index]) for index in self.active_indices
        }
        stable_indices = [
            index
            for index in self.active_indices
            if slopes[index] is not None
            and abs(slopes[index]) <= self.slope_threshold
        ]

        # N is calculated once from the entire post-update batch. Every normal
        # replacement, or every row in a newly started pool round, starts at N.
        replacement_rho = self.mean_rho()
        remaining = self.pool_size - self.next_pool_index
        round_restart = remaining == 0
        stable_replace_count = 0
        if round_restart:
            # All rows in the current round have already entered an active batch.
            # This update guarantees the last introduced rows were sampled at least
            # once. Start the next ordered round from its first full batch.
            replace_count = len(self.active_indices)
            self.active_indices = list(range(self.batch_size))
            self.states = {
                index: SampleNoiseState(rho=replacement_rho)
                for index in self.active_indices
            }
            self.next_pool_index = self.batch_size
            self.retired_total += replace_count
            self.samples_introduced_total += self.batch_size
            self.rounds_completed += 1
            self.pool_round += 1
            unfilled_retirements = 0
        else:
            stable_replace_count = min(len(stable_indices), remaining)
            replace_count = stable_replace_count
            replacement_indices = list(
                range(
                    self.next_pool_index,
                    self.next_pool_index + stable_replace_count,
                )
            )
            replacement_map = dict(
                zip(stable_indices[:stable_replace_count], replacement_indices)
            )

            next_active: list[int] = []
            for index in self.active_indices:
                replacement_index = replacement_map.get(index)
                if replacement_index is None:
                    next_active.append(index)
                    continue
                del self.states[index]
                self.states[replacement_index] = SampleNoiseState(rho=replacement_rho)
                next_active.append(replacement_index)
            self.active_indices = next_active
            self.next_pool_index += stable_replace_count
            self.retired_total += stable_replace_count
            self.samples_introduced_total += stable_replace_count
            # These candidates remain active until the last newly introduced rows
            # have trained once and the next call starts a fresh pool round.
            unfilled_retirements = len(stable_indices) - stable_replace_count

        active_rhos = [self.states[index].rho for index in self.active_indices]
        finite_slopes = [value for value in slopes.values() if value is not None]
        metrics = {
            "denoise/v2/active_batch_size": float(len(self.active_indices)),
            "denoise/v2/pool_size": float(self.pool_size),
            "denoise/v2/pool_cursor": float(self.next_pool_index),
            "denoise/v2/covered_this_round": float(self.next_pool_index),
            "denoise/v2/coverage_fraction_this_round": float(
                self.next_pool_index / self.pool_size
            ),
            "denoise/v2/pool_remaining": float(self.pool_size - self.next_pool_index),
            "denoise/v2/pool_round": float(self.pool_round),
            "denoise/v2/rounds_completed": float(self.rounds_completed),
            "denoise/v2/round_restart_this_step": float(round_restart),
            "denoise/v2/samples_introduced_total": float(
                self.samples_introduced_total
            ),
            "denoise/v2/accuracy_mean": float(np.mean(average_accuracies)),
            "denoise/v2/rho_used_mean": float(np.mean(used_rhos)),
            "denoise/v2/rho_replacement": replacement_rho,
            "denoise/v2/rho_active_mean": float(np.mean(active_rhos)),
            "denoise/v2/rho_active_min": float(np.min(active_rhos)),
            "denoise/v2/rho_active_max": float(np.max(active_rhos)),
            "denoise/v2/stable_candidates": float(len(stable_indices)),
            "denoise/v2/slope_abs_threshold": self.slope_threshold,
            "denoise/v2/replaced_this_step": float(replace_count),
            "denoise/v2/stable_replaced_this_step": float(stable_replace_count),
            "denoise/v2/retired_total": float(self.retired_total),
            "denoise/v2/unfilled_retirements": float(unfilled_retirements),
            "denoise/v2/pool_cycles_enabled": 1.0,
            "denoise/v2/slopes_defined": float(len(finite_slopes)),
        }
        if finite_slopes:
            absolute_slopes = [abs(value) for value in finite_slopes]
            metrics.update(
                {
                    "denoise/v2/slope_mean": float(np.mean(finite_slopes)),
                    "denoise/v2/slope_min": float(np.min(finite_slopes)),
                    "denoise/v2/slope_max": float(np.max(finite_slopes)),
                    "denoise/v2/slope_abs_mean": float(np.mean(absolute_slopes)),
                    "denoise/v2/slope_abs_min": float(np.min(absolute_slopes)),
                    "denoise/v2/slope_abs_max": float(np.max(absolute_slopes)),
                }
            )
        return metrics

    def metrics(self) -> dict[str, float]:
        active_rhos = [self.states[index].rho for index in self.active_indices]
        return {
            "denoise/v2/enabled": 1.0,
            "denoise/v2/active_batch_size": float(len(self.active_indices)),
            "denoise/v2/pool_size": float(self.pool_size),
            "denoise/v2/pool_cursor": float(self.next_pool_index),
            "denoise/v2/pool_remaining": float(self.pool_size - self.next_pool_index),
            "denoise/v2/coverage_fraction_this_round": float(
                self.next_pool_index / self.pool_size
            ),
            "denoise/v2/pool_round": float(self.pool_round),
            "denoise/v2/rounds_completed": float(self.rounds_completed),
            "denoise/v2/samples_introduced_total": float(
                self.samples_introduced_total
            ),
            "denoise/v2/rho_active_mean": float(np.mean(active_rhos)),
            "denoise/v2/rho_active_min": float(np.min(active_rhos)),
            "denoise/v2/rho_active_max": float(np.max(active_rhos)),
        }

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "pool_size": self.pool_size,
            "problem_id_reprs": [repr(problem_id) for problem_id in self.problem_ids],
            "batch_size": self.batch_size,
            "active_indices": list(self.active_indices),
            "next_pool_index": self.next_pool_index,
            "retired_total": self.retired_total,
            "pool_round": self.pool_round,
            "rounds_completed": self.rounds_completed,
            "samples_introduced_total": self.samples_introduced_total,
            "states": {
                str(index): {
                    "rho": state.rho,
                    "history": list(state.history),
                    "num_samples": state.num_samples,
                }
                for index, state in self.states.items()
            },
        }

    def load_state_dict(self, saved: Mapping) -> None:
        if int(saved.get("version", -1)) != self.STATE_VERSION:
            raise ValueError("Unsupported DenoiseRL v2 curriculum state version.")
        if int(saved.get("pool_size", -1)) != self.pool_size:
            raise ValueError("Curriculum checkpoint pool size does not match the dataset.")
        if saved.get("problem_id_reprs") != [repr(pid) for pid in self.problem_ids]:
            raise ValueError("Curriculum checkpoint pool order does not match the dataset.")
        if int(saved.get("batch_size", -1)) != self.batch_size:
            raise ValueError("Curriculum checkpoint batch size does not match the config.")

        active_indices = [int(index) for index in saved["active_indices"]]
        if len(active_indices) != self.batch_size or len(set(active_indices)) != self.batch_size:
            raise ValueError("Invalid active batch in curriculum checkpoint.")
        if any(index < 0 or index >= self.pool_size for index in active_indices):
            raise ValueError("Out-of-range active index in curriculum checkpoint.")

        states: dict[int, SampleNoiseState] = {}
        for index in active_indices:
            raw_state = saved["states"][str(index)]
            rho = self._finite("checkpoint rho", raw_state["rho"])
            if not (self.min_rho <= rho <= self.max_rho):
                raise ValueError("Checkpoint rho is outside the configured bounds.")
            history = [
                self._finite("checkpoint rho history", rho_value)
                for rho_value in raw_state["history"]
            ][-self.history_window :]
            states[index] = SampleNoiseState(
                rho=rho,
                history=history,
                num_samples=int(raw_state["num_samples"]),
            )

        next_pool_index = int(saved["next_pool_index"])
        if not (self.batch_size <= next_pool_index <= self.pool_size):
            raise ValueError("Invalid pool cursor in curriculum checkpoint.")
        self.active_indices = active_indices
        self.next_pool_index = next_pool_index
        self.retired_total = int(saved["retired_total"])
        self.pool_round = int(saved["pool_round"])
        self.rounds_completed = int(saved["rounds_completed"])
        self.samples_introduced_total = int(saved["samples_introduced_total"])
        self.states = states
