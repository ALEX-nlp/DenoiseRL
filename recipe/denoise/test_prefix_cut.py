import os
import subprocess
import unittest
from pathlib import Path

from recipe.denoise.prefix_cut import cut_wrong_solution_prefix


class CharacterTokenizer:
    """A minimal reversible tokenizer: one Unicode character per token."""

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


class PrefixCutTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_token_strategy_preserves_legacy_floor_cut(self):
        result = cut_wrong_solution_prefix(self.tokenizer, "abcdefghij", 0.29, "token")

        self.assertEqual(result.text, "ab")
        self.assertEqual(result.token_count, 2)
        self.assertFalse(result.used_line_boundary)

    def test_line_strategy_rounds_to_nearest_complete_line(self):
        text = "aaaa\nbbbbbbbbbb\ncc\n"

        first_line = cut_wrong_solution_prefix(self.tokenizer, text, 0.2, "line")
        second_line = cut_wrong_solution_prefix(self.tokenizer, text, 0.7, "line")

        self.assertEqual(first_line.text, "aaaa\n")
        self.assertEqual(second_line.text, "aaaa\nbbbbbbbbbb\n")
        self.assertTrue(first_line.used_line_boundary)
        self.assertTrue(second_line.used_line_boundary)

    def test_line_strategy_breaks_equal_distance_tie_toward_shorter_prefix(self):
        text = "aaaa\nbbbbbbbbb\n"

        result = cut_wrong_solution_prefix(self.tokenizer, text, 10 / 15, "line")

        self.assertEqual(result.text, "aaaa\n")
        self.assertEqual(result.token_count, 5)

    def test_line_strategy_falls_back_to_token_cut_for_single_line(self):
        result = cut_wrong_solution_prefix(self.tokenizer, "abcdefghij", 0.2, "line")

        self.assertEqual(result.text, "ab")
        self.assertFalse(result.used_line_boundary)
        self.assertAlmostEqual(result.realized_ratio, 0.2)

    def test_line_strategy_preserves_windows_line_ending(self):
        text = "abcd\r\nefghijkl\r\n"

        result = cut_wrong_solution_prefix(self.tokenizer, text, 0.3, "line")

        self.assertEqual(result.text, "abcd\r\n")
        self.assertTrue(result.used_line_boundary)

    def test_rejects_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "must be 'token' or 'line'"):
            cut_wrong_solution_prefix(self.tokenizer, "abc\ndef", 0.2, "sentence")


class LinePrefixScriptTest(unittest.TestCase):
    def test_wrapper_passes_line_cut_and_fixed_point_two_ratio(self):
        script_path = Path(__file__).with_name(
            "grpo_denoise_line_rho_qwen3-4b_v1.0.sh"
        )
        env = os.environ.copy()
        env.pop("part_response_ratio_strategy", None)
        env.pop("part_response_ratio_fixed", None)
        env.pop("partial_wrong_cut_strategy", None)
        result = subprocess.run(
            [
                "bash",
                "-c",
                'python3() { printf "%s\\n" "$@"; }; export -f python3; '
                'script="$1"; set --; source "$script"',
                "bash",
                str(script_path),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("+trainer.partial_wrong_cut_strategy=line", args)
        self.assertIn("+trainer.part_response_ratio_strategy=fixed", args)
        self.assertIn("+trainer.part_response_ratio_fixed=0.2", args)
        experiment_arg = next(
            arg for arg in args if arg.startswith("trainer.experiment_name=")
        )
        self.assertIn("ratio-fix0.2-cut-line", experiment_arg)


if __name__ == "__main__":
    unittest.main()
