import unittest

from recipe.correct_prefix.per_sample_curriculum import (
    PerSampleCorrectPrefixCurriculum,
    has_usable_correct_solution,
)


class CorrectPrefixCurriculumTest(unittest.TestCase):
    def test_pool_filter_accepts_only_non_empty_correct_rollouts(self):
        self.assertFalse(has_usable_correct_solution(None))
        self.assertFalse(has_usable_correct_solution([]))
        self.assertFalse(has_usable_correct_solution(["", "  "]))
        self.assertTrue(has_usable_correct_solution("verified solution"))
        self.assertTrue(has_usable_correct_solution(["", "verified solution"]))

    def test_above_target_accuracy_shortens_prefix(self):
        curriculum = PerSampleCorrectPrefixCurriculum(
            [10],
            batch_size=1,
            initial_rho=0.8,
            min_rho=0.0,
            max_rho=0.8,
            target_accuracy=0.75,
            alpha=0.2,
        )

        curriculum.update({10: 1.0})

        self.assertAlmostEqual(curriculum.rho_for_problem(10), 0.75)

    def test_below_target_accuracy_lengthens_prefix(self):
        curriculum = PerSampleCorrectPrefixCurriculum(
            [10],
            batch_size=1,
            initial_rho=0.4,
            min_rho=0.0,
            max_rho=0.8,
            target_accuracy=0.75,
            alpha=0.2,
        )

        curriculum.update({10: 0.25})

        self.assertAlmostEqual(curriculum.rho_for_problem(10), 0.5)

    def test_reverse_updates_clip_to_bounds(self):
        upper = PerSampleCorrectPrefixCurriculum(
            [10],
            batch_size=1,
            initial_rho=0.79,
            min_rho=0.2,
            max_rho=0.8,
            target_accuracy=1.0,
            alpha=1.0,
        )
        lower = PerSampleCorrectPrefixCurriculum(
            [20],
            batch_size=1,
            initial_rho=0.21,
            min_rho=0.2,
            max_rho=0.8,
            target_accuracy=0.0,
            alpha=1.0,
        )

        upper.update({10: 0.0})
        lower.update({20: 1.0})

        self.assertEqual(upper.rho_for_problem(10), 0.8)
        self.assertEqual(lower.rho_for_problem(20), 0.2)


if __name__ == "__main__":
    unittest.main()
