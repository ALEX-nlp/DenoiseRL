import unittest

import torch

from recipe.denoise_v2.length_reward import (
    apply_correct_length_reward,
    linear_length_factor,
)


class LinearLengthFactorTest(unittest.TestCase):
    def test_linear_endpoints_and_midpoint(self):
        factors = linear_length_factor(
            torch.tensor([0, 50, 100]),
            max_response_length=100,
            min_factor=0.5,
        )

        torch.testing.assert_close(factors, torch.tensor([1.0, 0.75, 0.5]))

    def test_lengths_above_budget_are_clamped(self):
        factors = linear_length_factor(
            torch.tensor([100, 150]),
            max_response_length=100,
            min_factor=0.5,
        )

        torch.testing.assert_close(factors, torch.tensor([0.5, 0.5]))

    def test_rejects_negative_lengths(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            linear_length_factor(
                torch.tensor([-1]),
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
            response_lengths=torch.tensor([0, 100, 100]),
            max_response_length=100,
            min_factor=0.5,
        )

        torch.testing.assert_close(
            shaped,
            torch.tensor(
                [
                    [0.0, 1.0],
                    [0.0, 0.5],
                    [0.0, -0.5],
                ]
            ),
        )
        torch.testing.assert_close(
            effective_factors, torch.tensor([1.0, 0.5, 1.0])
        )

    def test_requires_one_length_and_correctness_value_per_row(self):
        with self.assertRaisesRegex(ValueError, "one value per reward row"):
            apply_correct_length_reward(
                torch.zeros(2, 4),
                correctness=torch.tensor([1.0]),
                response_lengths=torch.tensor([1, 2]),
                max_response_length=4,
            )


if __name__ == "__main__":
    unittest.main()
