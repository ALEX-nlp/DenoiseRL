import unittest

from recipe.denoise_v2.per_sample_curriculum import (
    PerSampleNoiseCurriculum,
    has_usable_wrong_solution,
)


class PerSampleNoiseCurriculumTest(unittest.TestCase):
    def test_pool_filter_removes_unusable_rows_and_preserves_order(self):
        rows = [
            (10, []),
            (11, ["  ", "wrong-11"]),
            (12, None),
            (13, "wrong-13"),
            (14, [""]),
            (15, ["wrong-15"]),
        ]
        filtered_problem_ids = [
            problem_id
            for problem_id, wrong_solutions in rows
            if has_usable_wrong_solution(wrong_solutions)
        ]

        self.assertEqual(filtered_problem_ids, [11, 13, 15])
        curriculum = PerSampleNoiseCurriculum(filtered_problem_ids, batch_size=2)
        self.assertEqual(curriculum.active_problem_ids, (11, 13))

    def test_initial_active_batch_is_first_pool_slice_at_zero_rho(self):
        curriculum = PerSampleNoiseCurriculum([10, 11, 12, 13], batch_size=2)

        self.assertEqual(curriculum.active_indices, [0, 1])
        self.assertEqual(curriculum.active_problem_ids, (10, 11))
        self.assertEqual(curriculum.rho_for_problem(10), 0.0)
        self.assertEqual(curriculum.rho_for_problem(11), 0.0)

    def test_each_problem_updates_from_its_own_average_accuracy(self):
        curriculum = PerSampleNoiseCurriculum(
            [10, 11, 12],
            batch_size=2,
            target_accuracy=0.5,
            alpha=0.1,
        )

        curriculum.update({10: 1.0, 11: 0.0})

        self.assertAlmostEqual(curriculum.rho_for_problem(10), 0.05)
        self.assertEqual(curriculum.rho_for_problem(11), 0.0)

    def test_default_alpha_is_point_two(self):
        curriculum = PerSampleNoiseCurriculum(
            [10, 11], batch_size=1, target_accuracy=0.5
        )

        curriculum.update({10: 1.0})

        self.assertAlmostEqual(curriculum.rho_for_problem(10), 0.1)

    def test_stable_samples_get_consecutive_replacements_at_shared_mean_rho(self):
        curriculum = PerSampleNoiseCurriculum(
            [10, 11, 12, 13, 14],
            batch_size=3,
            target_accuracy=0.5,
            alpha=0.1,
        )
        # First sample only records one history point, so slope is not defined yet.
        curriculum.update({10: 1.0, 11: 0.5, 12: 0.0})

        metrics = curriculum.update({10: 1.0, 11: 0.5, 12: 0.0})

        # Used-rho histories are [0, .05], [0, 0], [0, 0]. The latter two
        # are inside the absolute-slope stability band and are replaced by
        # consecutive pool rows 3 and 4.
        self.assertEqual(curriculum.active_indices, [0, 3, 4])
        self.assertEqual(curriculum.active_problem_ids, (10, 13, 14))
        self.assertEqual(metrics["denoise/v2/replaced_this_step"], 2.0)
        expected_n = (0.1 + 0.0 + 0.0) / 3.0
        self.assertAlmostEqual(curriculum.rho_for_problem(13), expected_n)
        self.assertAlmostEqual(curriculum.rho_for_problem(14), expected_n)

    def test_large_negative_slope_is_not_stable(self):
        curriculum = PerSampleNoiseCurriculum(
            [10, 11],
            batch_size=1,
            initial_rho=0.3,
            target_accuracy=1.0,
            alpha=0.1,
            slope_threshold=0.0075,
        )
        curriculum.update({10: 0.0})  # used rho: 0.3
        metrics = curriculum.update({10: 0.0})  # used rho: 0.2, slope=-0.1

        self.assertEqual(curriculum.active_problem_ids, (10,))
        self.assertEqual(metrics["denoise/v2/stable_candidates"], 0.0)
        self.assertAlmostEqual(metrics["denoise/v2/slope_abs_mean"], 0.1)

    def test_small_positive_and_negative_slopes_are_stable(self):
        decreasing = PerSampleNoiseCurriculum(
            [10, 11],
            batch_size=1,
            initial_rho=0.3,
            target_accuracy=1.0,
            alpha=0.01,
            slope_threshold=0.0075,
        )
        decreasing.update({10: 0.0})
        decreasing.update({10: 0.0})  # slope=-0.01
        self.assertEqual(decreasing.active_problem_ids, (11,))

        increasing = PerSampleNoiseCurriculum(
            [20, 21],
            batch_size=1,
            target_accuracy=0.0,
            alpha=0.01,
            slope_threshold=0.0075,
        )
        increasing.update({20: 1.0})
        increasing.update({20: 1.0})  # slope=+0.01
        self.assertEqual(increasing.active_problem_ids, (21,))

    def test_slope_uses_only_recent_min_sample_count_and_window(self):
        curriculum = PerSampleNoiseCurriculum(
            [10, 11],
            batch_size=1,
            target_accuracy=0.0,
            alpha=0.1,
            history_window=3,
            min_history=3,
        )
        curriculum.update({10: 1.0})  # used: 0.0
        curriculum.update({10: 1.0})  # used: 0.1
        curriculum.update({10: 1.0})  # used: 0.2; positive slope
        self.assertEqual(curriculum.active_problem_ids, (10,))

        curriculum.update({10: 0.0})  # recent used: 0.1, 0.2, 0.3
        curriculum.update({10: 0.0})  # recent used: 0.2, 0.3, 0.3
        curriculum.update({10: 0.0})  # recent used: 0.3, 0.3, 0.3
        self.assertEqual(curriculum.active_problem_ids, (11,))

    def test_checkpoint_round_trip_preserves_active_batch_and_history(self):
        curriculum = PerSampleNoiseCurriculum(
            [10, 11, 12], batch_size=2, target_accuracy=0.5
        )
        curriculum.update({10: 1.0, 11: 0.0})
        state = curriculum.state_dict()

        restored = PerSampleNoiseCurriculum(
            [10, 11, 12], batch_size=2, target_accuracy=0.5
        )
        restored.load_state_dict(state)

        self.assertEqual(restored.state_dict(), state)

    def test_exhausted_pool_starts_a_new_ordered_round(self):
        curriculum = PerSampleNoiseCurriculum(
            [10, 11, 12], batch_size=2, target_accuracy=0.5
        )
        curriculum.update({10: 0.0, 11: 0.0})
        fill_metrics = curriculum.update({10: 0.0, 11: 0.0})

        # Only one unseen row remains. It fills one stable slot and must train at
        # least once before the pool advances to its next round.
        self.assertEqual(curriculum.active_problem_ids, (12, 11))
        self.assertEqual(fill_metrics["denoise/v2/pool_remaining"], 0.0)
        self.assertEqual(fill_metrics["denoise/v2/round_restart_this_step"], 0.0)

        restart_metrics = curriculum.update({12: 0.0, 11: 0.0})

        self.assertEqual(curriculum.active_problem_ids, (10, 11))
        self.assertEqual(curriculum.pool_round, 2)
        self.assertEqual(restart_metrics["denoise/v2/round_restart_this_step"], 1.0)
        self.assertEqual(restart_metrics["denoise/v2/rounds_completed"], 1.0)
        self.assertEqual(restart_metrics["denoise/v2/samples_introduced_total"], 5.0)


if __name__ == "__main__":
    unittest.main()
