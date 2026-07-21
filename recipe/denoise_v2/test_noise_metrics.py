import unittest

import numpy as np

from recipe.denoise_v2.noise_metrics import compute_paired_noise_acc_metrics


class _Batch:
    def __init__(self, **non_tensor_batch):
        self.non_tensor_batch = non_tensor_batch


class PairedNoiseAccMetricsTest(unittest.TestCase):
    def test_computes_problem_macro_averaged_noise_metrics(self):
        batch = _Batch(
            problem_id=np.array([1, 1, 1, 1, 2, 2, 2, 2]),
            partial_response_len=np.array([0, 0, 2, 2, 0, 0, 3, 3]),
            acc=np.array([1, 1, 1, 0, 0, 1, 0, 0], dtype=np.float32),
        )

        metrics = compute_paired_noise_acc_metrics(batch)

        self.assertEqual(metrics["reward_model/noise/n_paired_problems"], 2.0)
        self.assertEqual(metrics["reward_model/noise/paired_problem_coverage"], 1.0)
        self.assertAlmostEqual(metrics["reward_model/noise/paired_acc_clean"], 0.75)
        self.assertAlmostEqual(metrics["reward_model/noise/paired_acc_noise"], 0.25)
        self.assertAlmostEqual(metrics["reward_model/noise/paired_penalty"], 0.5)
        self.assertAlmostEqual(metrics["reward_model/noise/harm_rate"], 0.5)
        self.assertAlmostEqual(metrics["reward_model/noise/rescue_rate"], 0.0)
        self.assertAlmostEqual(metrics["reward_model/noise/sensitivity"], 0.5)
        self.assertAlmostEqual(
            metrics["reward_model/noise/recovery_given_clean"], 1.0 / 3.0
        )
        self.assertAlmostEqual(
            metrics["reward_model/noise/failure_given_clean"], 2.0 / 3.0
        )
        self.assertEqual(metrics["reward_model/noise/recovery_given_clean_defined"], 1.0)

    def test_weights_problems_equally_despite_different_rollout_counts(self):
        batch = _Batch(
            problem_id=np.array([1, 1, 1, 1, 2, 2, 2, 2]),
            partial_response_len=np.array([0, 0, 0, 2, 0, 2, 2, 2]),
            acc=np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32),
        )

        metrics = compute_paired_noise_acc_metrics(batch)

        self.assertEqual(metrics["reward_model/noise/paired_acc_clean"], 0.5)
        self.assertEqual(metrics["reward_model/noise/paired_acc_noise"], 0.5)
        self.assertEqual(metrics["reward_model/noise/paired_penalty"], 0.0)

    def test_reports_rescue_and_omits_undefined_conditional_recovery(self):
        batch = _Batch(
            problem_id=np.array([1, 1]),
            partial_response_len=np.array([0, 2]),
            acc=np.array([0, 1], dtype=np.float32),
        )

        metrics = compute_paired_noise_acc_metrics(batch)

        self.assertEqual(metrics["reward_model/noise/paired_penalty"], -1.0)
        self.assertEqual(metrics["reward_model/noise/harm_rate"], 0.0)
        self.assertEqual(metrics["reward_model/noise/rescue_rate"], 1.0)
        self.assertEqual(metrics["reward_model/noise/sensitivity"], 1.0)
        self.assertEqual(metrics["reward_model/noise/recovery_given_clean_defined"], 0.0)
        self.assertNotIn("reward_model/noise/recovery_given_clean", metrics)
        self.assertNotIn("reward_model/noise/failure_given_clean", metrics)

    def test_reports_pairing_coverage_and_ignores_invalid_rows(self):
        batch = _Batch(
            problem_id=np.array(
                ["paired", "paired", "clean", "noise", None, "nan", "invalid"]
            ),
            partial_response_len=np.array([0, 2, 0, 2, 0, 0, 0]),
            acc=np.array([1, 0, 1, 1, 1, np.nan, 2], dtype=np.float32),
        )

        metrics = compute_paired_noise_acc_metrics(batch)

        self.assertEqual(metrics["reward_model/noise/n_total_problems"], 3.0)
        self.assertEqual(metrics["reward_model/noise/n_paired_problems"], 1.0)
        self.assertEqual(metrics["reward_model/noise/n_clean_only_problems"], 1.0)
        self.assertEqual(metrics["reward_model/noise/n_noise_only_problems"], 1.0)
        self.assertAlmostEqual(
            metrics["reward_model/noise/paired_problem_coverage"], 1.0 / 3.0
        )

    def test_returns_no_metrics_when_pairing_fields_are_missing(self):
        self.assertEqual(compute_paired_noise_acc_metrics(_Batch(acc=[1.0])), {})


if __name__ == "__main__":
    unittest.main()
