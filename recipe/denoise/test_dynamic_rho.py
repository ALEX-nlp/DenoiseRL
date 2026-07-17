import unittest

from recipe.denoise.dynamic_rho import DynamicRhoController


class DynamicRhoControllerTest(unittest.TestCase):
    def test_increases_rho_when_recoverability_is_above_target(self):
        controller = DynamicRhoController(
            initial_rho=0.2, target_recoverability=0.8, alpha=0.05
        )

        metrics = controller.update_from_acc(acc_base=0.5, acc_noise=0.5)

        self.assertAlmostEqual(controller.current_rho, 0.21)
        self.assertAlmostEqual(metrics["denoise/dynamic_rho/recoverability"], 1.0)

    def test_decreases_rho_when_recoverability_is_below_target(self):
        controller = DynamicRhoController(
            initial_rho=0.2, target_recoverability=0.8, alpha=0.05
        )

        metrics = controller.update_from_acc(acc_base=0.5, acc_noise=0.25)

        self.assertAlmostEqual(controller.current_rho, 0.185)
        self.assertAlmostEqual(metrics["denoise/dynamic_rho/recoverability"], 0.5)

    def test_caps_recoverability_at_one(self):
        controller = DynamicRhoController(
            initial_rho=0.2, target_recoverability=0.8, alpha=0.05
        )

        metrics = controller.update_from_acc(acc_base=0.25, acc_noise=0.5)

        self.assertAlmostEqual(controller.current_rho, 0.21)
        self.assertAlmostEqual(metrics["denoise/dynamic_rho/recoverability"], 1.0)

    def test_skips_update_when_base_accuracy_is_zero(self):
        controller = DynamicRhoController(initial_rho=0.2)

        metrics = controller.update_from_acc(acc_base=0.0, acc_noise=0.0)

        self.assertAlmostEqual(controller.current_rho, 0.2)
        self.assertEqual(metrics["denoise/dynamic_rho/update_applied"], 0.0)
        self.assertEqual(metrics["denoise/dynamic_rho/update_skipped_zero_base"], 1.0)

    def test_accuracy_feedback_starts_from_zero_and_increases_above_target(self):
        controller = DynamicRhoController(
            min_rho=0.0,
            initial_rho=0.0,
            feedback=DynamicRhoController.ACCURACY,
            target_accuracy=0.75,
            alpha=0.05,
        )

        metrics = controller.update_from_metrics({"reward_model/acc": 0.95})

        self.assertAlmostEqual(controller.current_rho, 0.01)
        self.assertAlmostEqual(metrics["denoise/dynamic_rho/batch_accuracy"], 0.95)
        self.assertAlmostEqual(metrics["denoise/dynamic_rho/accuracy_error"], 0.2)

    def test_accuracy_feedback_decreases_immediately_below_target(self):
        controller = DynamicRhoController(
            min_rho=0.0,
            initial_rho=0.2,
            feedback=DynamicRhoController.ACCURACY,
            target_accuracy=0.75,
            alpha=0.05,
        )

        metrics = controller.update_from_accuracy(0.55)

        self.assertAlmostEqual(metrics["denoise/dynamic_rho/accuracy_error"], -0.2)
        self.assertAlmostEqual(controller.current_rho, 0.19)

    def test_accuracy_feedback_clamps_at_zero(self):
        controller = DynamicRhoController(
            min_rho=0.0,
            initial_rho=0.0,
            feedback=DynamicRhoController.ACCURACY,
            target_accuracy=0.75,
        )

        metrics = controller.update_from_accuracy(0.5)

        self.assertEqual(controller.current_rho, 0.0)
        self.assertEqual(metrics["denoise/dynamic_rho/rho_update_delta"], 0.0)

    def test_accuracy_feedback_does_not_require_base_or_noise_accuracy(self):
        controller = DynamicRhoController(
            min_rho=0.0,
            initial_rho=0.0,
            feedback=DynamicRhoController.ACCURACY,
        )

        metrics = controller.update_from_metrics({"reward_model/acc": 0.8})

        self.assertEqual(metrics["denoise/dynamic_rho/update_applied"], 1.0)
        self.assertEqual(metrics["denoise/dynamic_rho/feedback_is_accuracy"], 1.0)

    def test_accuracy_config_defaults_allow_zero_rho(self):
        controller = DynamicRhoController.from_trainer_config(
            {}, feedback=DynamicRhoController.ACCURACY
        )

        self.assertEqual(controller.min_rho, 0.0)
        self.assertEqual(controller.current_rho, 0.0)
        self.assertEqual(controller.target_accuracy, 0.75)


if __name__ == "__main__":
    unittest.main()
