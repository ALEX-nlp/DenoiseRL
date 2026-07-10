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


if __name__ == "__main__":
    unittest.main()
