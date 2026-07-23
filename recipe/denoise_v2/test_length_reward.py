import unittest

import torch

from recipe.denoise_v2.length_reward import (
    apply_correct_length_reward,
    dynamic_cutdown_length_factor,
)


class DynamicCutdownLengthFactorTest(unittest.TestCase):
    def test_prefix_defines_penalty_start_and_cache_width(self):
        factors = dynamic_cutdown_length_factor(
            torch.tensor([3000, 3296, 3500, 4096]),
            prefix_lengths=torch.tensor([800, 800, 800, 800]),
            max_response_length=4096,
        )

        torch.testing.assert_close(
            factors,
            torch.tensor([1.0, 1.0, 0.745, 0.0]),
        )

    def test_different_prefixes_create_different_dynamic_caches(self):
        factors = dynamic_cutdown_length_factor(
            torch.tensor([3584, 3840]),
            prefix_lengths=torch.tensor([1024, 512]),
            max_response_length=4096,
        )

        torch.testing.assert_close(factors, torch.tensor([0.5, 0.5]))

    def test_zero_prefix_has_no_length_penalty(self):
        factors = dynamic_cutdown_length_factor(
            torch.tensor([0, 4096]),
            prefix_lengths=torch.tensor([0, 0]),
            max_response_length=4096,
        )

        torch.testing.assert_close(factors, torch.tensor([1.0, 1.0]))

    def test_min_factor_is_respected_at_generation_limit(self):
        factors = dynamic_cutdown_length_factor(
            torch.tensor([100]),
            prefix_lengths=torch.tensor([20]),
            max_response_length=100,
            min_factor=0.5,
        )

        torch.testing.assert_close(factors, torch.tensor([0.5]))

    def test_rejects_negative_lengths(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            dynamic_cutdown_length_factor(
                torch.tensor([-1]),
                prefix_lengths=torch.tensor([20]),
                max_response_length=100,
            )

    def test_rejects_prefix_larger_than_response_budget(self):
        with self.assertRaisesRegex(ValueError, "prefix_lengths"):
            dynamic_cutdown_length_factor(
                torch.tensor([0]),
                prefix_lengths=torch.tensor([101]),
                max_response_length=100,
            )


class ApplyCorrectLengthRewardTest(unittest.TestCase):
    def test_only_correct_rollouts_are_scaled(self):
        rewards = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, -0.5],
            ]
        )

        shaped, effective_factors = apply_correct_length_reward(
            rewards,
            correctness=torch.tensor([1.0, 1.0, 0.0]),
            response_lengths=torch.tensor([3296, 4096, 4096]),
            prefix_lengths=torch.tensor([800, 800, 800]),
            max_response_length=4096,
        )

        torch.testing.assert_close(
            shaped,
            torch.tensor(
                [
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, -0.5],
                ]
            ),
        )
        torch.testing.assert_close(
            effective_factors, torch.tensor([1.0, 0.0, 1.0])
        )

    def test_requires_one_length_and_correctness_value_per_row(self):
        with self.assertRaisesRegex(ValueError, "one value per reward row"):
            apply_correct_length_reward(
                torch.zeros(2, 4),
                correctness=torch.tensor([1.0]),
                response_lengths=torch.tensor([1, 2]),
                prefix_lengths=torch.tensor([1, 2]),
                max_response_length=4,
            )

    def test_requires_one_prefix_length_per_row(self):
        with self.assertRaisesRegex(ValueError, "prefix_lengths"):
            apply_correct_length_reward(
                torch.zeros(2, 4),
                correctness=torch.tensor([1.0, 1.0]),
                response_lengths=torch.tensor([1, 2]),
                prefix_lengths=torch.tensor([1]),
                max_response_length=4,
            )


if __name__ == "__main__":
    unittest.main()
