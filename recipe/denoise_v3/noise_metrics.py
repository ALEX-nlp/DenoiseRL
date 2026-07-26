"""Metrics for measuring the effect of noisy prefixes on verifier accuracy."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


_METRIC_PREFIX = "reward_model/noise"


def _hashable_problem_id(value):
    """Return a hashable scalar problem id, or ``None`` for an invalid id."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        hash(value)
    except TypeError:
        return None
    return value


def compute_paired_noise_acc_metrics(batch) -> dict:
    """Compute problem-matched clean/noisy accuracy metrics.

    For every ``problem_id`` that has at least one finite clean accuracy
    (``partial_response_len == 0``) and one finite noisy accuracy
    (``partial_response_len > 0``), let ``c_i`` and ``n_i`` be the respective
    within-problem means.  Metrics are then macro-averaged across problems so
    problems with more rollout rows do not receive more weight.

    ``harm_rate`` and ``rescue_rate`` are expected clean-to-wrong and
    wrong-to-clean flip rates between independent clean/noisy draws for the same
    problem.  Consequently, ``paired_penalty == harm_rate - rescue_rate`` and
    ``sensitivity == harm_rate + rescue_rate`` (up to floating-point error).

    Missing required row-level fields return no metrics.  Invalid accuracies and
    missing/unhashable problem ids are ignored.  Conditional recovery is omitted
    when none of the paired problems has positive clean accuracy; its companion
    ``recovery_given_clean_defined`` metric is always emitted when paired input
    fields are available.
    """
    non_tensor_batch = getattr(batch, "non_tensor_batch", None)
    if non_tensor_batch is None:
        return {}

    acc_vals = non_tensor_batch.get("acc")
    problem_ids = non_tensor_batch.get("problem_id")
    partial_lens = non_tensor_batch.get("partial_response_len")
    if acc_vals is None or problem_ids is None or partial_lens is None:
        return {}

    acc_arr = np.asarray(acc_vals, dtype=np.float64).reshape(-1)
    problem_arr = np.asarray(problem_ids, dtype=object).reshape(-1)
    partial_arr = np.asarray(partial_lens).reshape(-1)
    n_rows = min(acc_arr.size, problem_arr.size, partial_arr.size)

    grouped = defaultdict(lambda: {"clean": [], "noise": []})
    for row_idx in range(n_rows):
        acc = acc_arr[row_idx]
        if not np.isfinite(acc) or not 0.0 <= acc <= 1.0:
            continue

        problem_id = _hashable_problem_id(problem_arr[row_idx])
        if problem_id is None:
            continue

        partial_len = partial_arr[row_idx]
        if partial_len == 0:
            grouped[problem_id]["clean"].append(float(acc))
        elif partial_len > 0:
            grouped[problem_id]["noise"].append(float(acc))

    paired_clean = []
    paired_noise = []
    n_clean_only = 0
    n_noise_only = 0
    for values in grouped.values():
        clean_vals = values["clean"]
        noise_vals = values["noise"]
        if clean_vals and noise_vals:
            paired_clean.append(float(np.mean(clean_vals)))
            paired_noise.append(float(np.mean(noise_vals)))
        elif clean_vals:
            n_clean_only += 1
        elif noise_vals:
            n_noise_only += 1

    n_total_problems = len(grouped)
    n_paired_problems = len(paired_clean)
    metrics = {
        f"{_METRIC_PREFIX}/n_total_problems": float(n_total_problems),
        f"{_METRIC_PREFIX}/n_paired_problems": float(n_paired_problems),
        f"{_METRIC_PREFIX}/n_clean_only_problems": float(n_clean_only),
        f"{_METRIC_PREFIX}/n_noise_only_problems": float(n_noise_only),
        f"{_METRIC_PREFIX}/recovery_given_clean_defined": 0.0,
    }
    if n_total_problems > 0:
        metrics[f"{_METRIC_PREFIX}/paired_problem_coverage"] = (
            float(n_paired_problems) / float(n_total_problems)
        )
    if n_paired_problems == 0:
        return metrics

    clean_arr = np.asarray(paired_clean, dtype=np.float64)
    noise_arr = np.asarray(paired_noise, dtype=np.float64)
    penalty_arr = clean_arr - noise_arr
    harm_arr = clean_arr * (1.0 - noise_arr)
    rescue_arr = (1.0 - clean_arr) * noise_arr

    metrics.update(
        {
            f"{_METRIC_PREFIX}/paired_acc_clean": float(np.mean(clean_arr)),
            f"{_METRIC_PREFIX}/paired_acc_noise": float(np.mean(noise_arr)),
            f"{_METRIC_PREFIX}/paired_penalty": float(np.mean(penalty_arr)),
            f"{_METRIC_PREFIX}/harm_rate": float(np.mean(harm_arr)),
            f"{_METRIC_PREFIX}/rescue_rate": float(np.mean(rescue_arr)),
            f"{_METRIC_PREFIX}/sensitivity": float(np.mean(harm_arr + rescue_arr)),
        }
    )

    clean_success_mass = float(np.sum(clean_arr))
    if clean_success_mass > 0.0:
        recovery = float(np.sum(clean_arr * noise_arr) / clean_success_mass)
        metrics.update(
            {
                f"{_METRIC_PREFIX}/recovery_given_clean": recovery,
                f"{_METRIC_PREFIX}/failure_given_clean": 1.0 - recovery,
                f"{_METRIC_PREFIX}/recovery_given_clean_defined": 1.0,
            }
        )

    return metrics
