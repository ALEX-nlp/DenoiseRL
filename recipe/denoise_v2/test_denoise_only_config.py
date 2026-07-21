import os
import subprocess
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("grpo_denoise_qwen3-4b_v2.0.sh")


class DenoiseV2ConfigTest(unittest.TestCase):
    def _expand_script(self, **overrides):
        env = os.environ.copy()
        env.update({key: str(value) for key, value in overrides.items()})
        return subprocess.run(
            [
                "bash",
                "-c",
                'python3() { printf "%s\\n" "$@"; }; export -f python3; source "$1"',
                "bash",
                str(SCRIPT_PATH),
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
        self.assertIn("+trainer.v2_curriculum_enabled=True", args)
        self.assertIn("+trainer.v2_initial_rho=0.0", args)
        self.assertIn("+trainer.v2_alpha=0.2", args)
        self.assertIn("+trainer.v2_history_window=10", args)
        self.assertIn("+trainer.v2_slope_threshold=0.0075", args)
        self.assertIn("trainer.total_epochs=10000", args)

    def test_exposes_accuracy_controller_values_as_environment_overrides(self):
        result = self._expand_script(v2_target_accuracy=0.6, v2_alpha=0.02)

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("+trainer.v2_target_accuracy=0.6", args)
        self.assertIn("+trainer.v2_alpha=0.02", args)


if __name__ == "__main__":
    unittest.main()
