import unittest

from recipe.denoise_v2.prefix_cut import cut_wrong_solution_prefix


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

if __name__ == "__main__":
    unittest.main()
