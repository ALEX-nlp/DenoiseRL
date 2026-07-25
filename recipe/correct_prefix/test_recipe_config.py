import re
import unittest
from pathlib import Path


RECIPE_DIR = Path(__file__).resolve().parent
SCRIPT_PATHS = (
    RECIPE_DIR / "grpo_correct_prefix_qwen3-4b_v1.0.sh",
    RECIPE_DIR / "grpo_correct_prefix_qwen3-8b_v1.0.sh",
)


class CorrectPrefixRecipeConfigTest(unittest.TestCase):
    def test_training_scripts_use_only_the_copied_recipe(self):
        for script_path in SCRIPT_PATHS:
            script = script_path.read_text(encoding="utf-8")
            self.assertIn(
                "python3 -m recipe.correct_prefix.main_dapo",
                script,
            )
            self.assertNotIn("recipe.denoise_v2", script)
            self.assertIn("+trainer.noise_source=partial_correct", script)
            self.assertIn("+trainer.correct_prefix_curriculum_enabled=True", script)
            self.assertIn("with_correct_boxed", script)

    def test_each_script_starts_at_its_maximum_prefix_ratio(self):
        pattern = re.compile(
            r"correct_prefix_(initial|max)_rho="
            r"\$\{correct_prefix_(?:initial|max)_rho:-(?P<value>[0-9.]+)\}"
        )
        for script_path in SCRIPT_PATHS:
            values = [
                float(match.group("value"))
                for match in pattern.finditer(
                    script_path.read_text(encoding="utf-8")
                )
            ]
            self.assertEqual(len(values), 2)
            self.assertEqual(values[0], values[1])

    def test_data_collector_targets_verified_correct_rollouts(self):
        source = (RECIPE_DIR / "data_prepare.py").read_text(encoding="utf-8")
        self.assertIn('df_out["correct_answer_with_boxed"]', source)
        self.assertIn("boxed is not None and acc >= 1.0", source)
        self.assertNotIn('df_out["wrong_answer_with_boxed"]', source)


if __name__ == "__main__":
    unittest.main()
