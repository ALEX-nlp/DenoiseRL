import unittest

from recipe.denoise_v3.random_token_noise import scaled_random_token_count


class ScaledRandomTokenCountTest(unittest.TestCase):
    def test_scales_maximum_by_rho(self):
        self.assertEqual(scaled_random_token_count(2048, 0.0), 0)
        self.assertEqual(scaled_random_token_count(2048, 0.25), 512)
        self.assertEqual(scaled_random_token_count(2048, 0.5), 1024)
        self.assertEqual(scaled_random_token_count(2048, 1.0), 2048)

    def test_fractional_count_rounds_down(self):
        self.assertEqual(scaled_random_token_count(3, 0.5), 1)

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "max_random_token"):
            scaled_random_token_count(0, 0.5)
        with self.assertRaisesRegex(ValueError, "rho"):
            scaled_random_token_count(2048, 1.1)
        with self.assertRaisesRegex(ValueError, "rho"):
            scaled_random_token_count(2048, float("nan"))


if __name__ == "__main__":
    unittest.main()
