import unittest

import torch

from recipe.denoise_v2.length_reward import (
    apply_dynamic_length_reward,
    apply_response_clip_penalty,
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


class ApplyDynamicLengthRewardTest(unittest.TestCase):
    def test_only_correct_rollouts_are_scaled(self):
        rewards = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, -0.5],
            ]
        )

        shaped, effective_factors, effective_penalties = apply_dynamic_length_reward(
            rewards,
            correctness=torch.tensor([1.0, 1.0, 0.0]),
            response_lengths=torch.tensor([3296, 4096, 4096]),
            prefix_lengths=torch.tensor([800, 800, 800]),
            reward_positions=torch.tensor([1, 1, 1]),
            max_response_length=4096,
            scope="correct",
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
        torch.testing.assert_close(
            effective_penalties, torch.tensor([0.0, 1.0, 0.0])
        )

    def test_requires_one_length_and_correctness_value_per_row(self):
        with self.assertRaisesRegex(ValueError, "one value per reward row"):
            apply_dynamic_length_reward(
                torch.zeros(2, 4),
                correctness=torch.tensor([1.0]),
                response_lengths=torch.tensor([1, 2]),
                prefix_lengths=torch.tensor([1, 2]),
                reward_positions=torch.tensor([0, 0]),
                max_response_length=4,
            )

    def test_requires_one_prefix_length_per_row(self):
        with self.assertRaisesRegex(ValueError, "prefix_lengths"):
            apply_dynamic_length_reward(
                torch.zeros(2, 4),
                correctness=torch.tensor([1.0, 1.0]),
                response_lengths=torch.tensor([1, 2]),
                prefix_lengths=torch.tensor([1]),
                reward_positions=torch.tensor([0, 0]),
                max_response_length=4,
            )

    def test_all_scope_gives_incorrect_rollouts_negative_reward(self):
        rewards = torch.zeros(2, 3)

        shaped, effective_factors, effective_penalties = apply_dynamic_length_reward(
            rewards,
            correctness=torch.tensor([0.0, 0.0]),
            response_lengths=torch.tensor([3296, 4096]),
            prefix_lengths=torch.tensor([800, 800]),
            reward_positions=torch.tensor([1, 2]),
            max_response_length=4096,
            scope="all",
        )

        torch.testing.assert_close(
            shaped,
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            ),
        )
        torch.testing.assert_close(
            effective_factors, torch.tensor([1.0, 0.0])
        )
        torch.testing.assert_close(
            effective_penalties, torch.tensor([0.0, 1.0])
        )

    def test_rejects_unknown_scope(self):
        with self.assertRaisesRegex(ValueError, "scope"):
            apply_dynamic_length_reward(
                torch.zeros(1, 2),
                correctness=torch.tensor([1.0]),
                response_lengths=torch.tensor([2]),
                prefix_lengths=torch.tensor([1]),
                reward_positions=torch.tensor([1]),
                max_response_length=2,
                scope="unknown",
            )


class ApplyResponseClipPenaltyTest(unittest.TestCase):
    def test_only_responses_at_generation_limit_are_penalized(self):
        rewards = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -0.5, 0.0],
            ]
        )

        shaped, clipped_mask = apply_response_clip_penalty(
            rewards,
            response_lengths=torch.tensor([3, 2, 3]),
            reward_positions=torch.tensor([1, 2, 1]),
            max_response_length=3,
            penalty=0.25,
        )

        torch.testing.assert_close(
            shaped,
            torch.tensor(
                [
                    [0.0, 0.75, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -0.75, 0.0],
                ]
            ),
        )
        torch.testing.assert_close(
            clipped_mask, torch.tensor([True, False, True])
        )

    def test_rejects_negative_penalty(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            apply_response_clip_penalty(
                torch.zeros(1, 2),
                response_lengths=torch.tensor([2]),
                reward_positions=torch.tensor([1]),
                max_response_length=2,
                penalty=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
