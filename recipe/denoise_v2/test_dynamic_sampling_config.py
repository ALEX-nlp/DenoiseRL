import os
import subprocess
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
SCRIPT_4B = SCRIPT_DIR / "grpo_denoise_dynamic_sample_line_qwen3-4b_v2.0.sh"
SCRIPT_8B = SCRIPT_DIR / "grpo_denoise_dynamic_sample_line_qwen3-8b_v2.0.sh"


class DynamicSamplingConfigTest(unittest.TestCase):
    def _expand_script(self, script_path, **overrides):
        env = os.environ.copy()
        env.update({key: str(value) for key, value in overrides.items()})
        return subprocess.run(
            [
                "bash",
                "-c",
                'python3() { printf "%s\\n" "$@"; }; export -f python3; "$1"',
                "bash",
                str(script_path),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_4b_enables_only_dapo_dynamic_sampling(self):
        result = self._expand_script(
            SCRIPT_4B,
            sample_prompt_bsz=24,
            backward_prompt_bsz=8,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("data.gen_batch_size=24", args)
        self.assertIn("data.train_batch_size=8", args)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=8", args)
        self.assertIn("+trainer.use_dapo=True", args)
        self.assertIn("+trainer.dapo_max_num_gen_batches=10", args)
        self.assertIn("algorithm.filter_groups.enable=False", args)
        self.assertIn("reward_model.reward_manager=naive", args)
        self.assertIn("reward_model.overlong_buffer.enable=False", args)

    def test_default_sampling_and_backward_prompt_counts_are_separate(self):
        result = self._expand_script(SCRIPT_4B)

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("data.gen_batch_size=48", args)
        self.assertIn("data.train_batch_size=16", args)

    def test_uses_line_prefix_none_mode_and_no_length_reward(self):
        result = self._expand_script(SCRIPT_4B)

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("+trainer.partial_wrong_cut_strategy=line", args)
        self.assertIn("+trainer.partial_mode=none", args)
        self.assertIn("+trainer.correct_length_reward_enabled=False", args)
        self.assertIn("+trainer.response_clip_reward_penalty=0.0", args)
        self.assertIn("reward_model.overlong_buffer.penalty_factor=0.0", args)

    def test_8b_wrapper_selects_the_8b_model_and_project(self):
        result = self._expand_script(SCRIPT_8B)

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn(
            "actor_rollout_ref.model.path=../Model/Qwen/Qwen3-8B-Base", args
        )
        self.assertIn("trainer.project_name=DenoiseRL-v2-8B", args)


if __name__ == "__main__":
    unittest.main()
