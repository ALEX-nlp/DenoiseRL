import os
import subprocess
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("grpo_denoise_qwen3-4b_v3.0.sh")
SCRIPT_8B_PATH = Path(__file__).with_name("grpo_denoise_qwen3-8b_v3.0.sh")
RANDOM_TOKEN_SCRIPT_PATH = Path(__file__).with_name(
    "grpo_denoise_random_tokens_qwen3-4b_v3.0.sh"
)


class DenoiseV3ConfigTest(unittest.TestCase):
    def _expand_script(self, script_path=SCRIPT_PATH, **overrides):
        env = os.environ.copy()
        env.update({key: str(value) for key, value in overrides.items()})
        return subprocess.run(
            [
                "bash",
                "-c",
                'python3() { printf "%s\\n" "$@"; }; export -f python3; source "$1"',
                "bash",
                str(script_path),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_uses_exactly_sixteen_noise_rollouts(self):
        result = self._expand_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("actor_rollout_ref.rollout.n=0", result.stdout.splitlines())
        self.assertIn("actor_rollout_ref.actor.rollout_n=16", result.stdout.splitlines())
        self.assertIn("+trainer.sub_rollout_k=16", result.stdout.splitlines())

    def test_enables_ordered_per_sample_curriculum(self):
        result = self._expand_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("data.shuffle=False", args)
        self.assertIn("+trainer.part_response_ratio_strategy=fixed", args)
        self.assertIn("+trainer.partial_mode=none", args)
        self.assertIn("+trainer.v2_curriculum_enabled=True", args)
        self.assertIn("+trainer.v2_initial_rho=0.0", args)
        self.assertIn("+trainer.v2_target_accuracy=0.75", args)
        self.assertIn("+trainer.v2_alpha=0.2", args)
        self.assertIn("+trainer.v2_history_window=5", args)
        self.assertIn("+trainer.v2_slope_threshold=0.02", args)
        self.assertIn("trainer.total_epochs=10000", args)

    def test_exposes_accuracy_controller_values_as_environment_overrides(self):
        result = self._expand_script(v2_target_accuracy=0.6, v2_alpha=0.02)

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("+trainer.v2_target_accuracy=0.6", args)
        self.assertIn("+trainer.v2_alpha=0.02", args)

    def test_enables_first_correct_box_reward_without_old_length_shaping(self):
        result = self._expand_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("+trainer.correct_length_reward_enabled=False", args)
        self.assertIn("+trainer.correct_length_reward_min_factor=0.0", args)
        self.assertIn("+trainer.length_reward_scope=all", args)
        self.assertIn("+trainer.first_correct_box_reward_enabled=True", args)
        self.assertIn("+trainer.first_correct_box_max_boxes=10", args)
        self.assertIn("+trainer.first_correct_box_erd_round_digits=2", args)

    def test_8b_uses_first_correct_box_reward_by_default(self):
        result = self._expand_script(SCRIPT_8B_PATH)

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("+trainer.correct_length_reward_enabled=False", args)
        self.assertIn("+trainer.correct_length_reward_min_factor=0.0", args)
        self.assertIn("+trainer.length_reward_scope=all", args)
        self.assertIn("+trainer.first_correct_box_reward_enabled=True", args)
        self.assertIn("+trainer.first_correct_box_max_boxes=10", args)

    def test_exposes_length_reward_as_environment_overrides(self):
        result = self._expand_script(
            correct_length_reward_enabled=False,
            correct_length_reward_min_factor=0.7,
            length_reward_scope="correct",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("+trainer.correct_length_reward_enabled=False", args)
        self.assertIn("+trainer.correct_length_reward_min_factor=0.7", args)
        self.assertIn("+trainer.length_reward_scope=correct", args)

    def test_first_correct_box_reward_is_optional_and_overridable(self):
        result = self._expand_script(
            first_correct_box_reward_enabled=False,
            first_correct_box_max_boxes=7,
            first_correct_box_erd_round_digits=3,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("+trainer.first_correct_box_reward_enabled=False", args)
        self.assertIn("+trainer.first_correct_box_max_boxes=7", args)
        self.assertIn("+trainer.first_correct_box_erd_round_digits=3", args)

    def test_random_token_entrypoint_enables_dynamic_maximum(self):
        result = self._expand_script(RANDOM_TOKEN_SCRIPT_PATH)

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("+trainer.noise_source=random_tokens", args)
        self.assertIn("+trainer.max_random_token=2048", args)
        self.assertIn("+trainer.random_noise_exclude_special=True", args)
        self.assertIn("+trainer.v2_curriculum_enabled=True", args)
        self.assertIn("+trainer.v2_initial_rho=0.0", args)
        self.assertIn("+trainer.v2_max_rho=1.0", args)

    def test_random_token_maximum_is_overridable(self):
        result = self._expand_script(
            RANDOM_TOKEN_SCRIPT_PATH,
            max_random_token=1024,
            v2_max_rho=0.75,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("+trainer.max_random_token=1024", args)
        self.assertIn("+trainer.v2_max_rho=0.75", args)


if __name__ == "__main__":
    unittest.main()
