import os
import subprocess
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("denoise_only_qwen3-4b_v1.0.sh")


class DenoiseOnlyConfigTest(unittest.TestCase):
    def _expand_script(self, **overrides):
        env = os.environ.copy()
        env.pop("n_resp_per_prompt", None)
        env.pop("sub_rollout_k", None)
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

    def test_default_uses_only_sub_rollouts_for_actor_batch_size(self):
        result = self._expand_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("actor_rollout_ref.rollout.n=0", result.stdout.splitlines())
        self.assertIn("actor_rollout_ref.actor.rollout_n=16", result.stdout.splitlines())
        self.assertIn("+trainer.sub_rollout_k=16", result.stdout.splitlines())

    def test_effective_actor_rollout_n_is_n_plus_k(self):
        result = self._expand_script(n_resp_per_prompt=2, sub_rollout_k=3)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("actor_rollout_ref.rollout.n=2", result.stdout.splitlines())
        self.assertIn("actor_rollout_ref.actor.rollout_n=5", result.stdout.splitlines())

    def test_rejects_zero_total_rollouts(self):
        result = self._expand_script(n_resp_per_prompt=0, sub_rollout_k=0)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("n_resp_per_prompt + sub_rollout_k must be > 0, got 0.", result.stderr)


if __name__ == "__main__":
    unittest.main()
