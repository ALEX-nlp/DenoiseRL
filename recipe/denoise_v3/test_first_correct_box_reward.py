import unittest

import torch

from recipe.denoise_v3.first_correct_box_reward import (
    analyze_first_correct_box,
    apply_first_correct_box_reward,
    find_complete_boxed_spans,
    split_response_at_boxed_ends,
)


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids if token_id != 0)


class FirstCorrectBoxParserTest(unittest.TestCase):
    def test_finds_balanced_boxes_with_nested_braces_and_whitespace(self):
        text = r"a \boxed {1+\frac{2}{3}} b \boxed{4}"

        boxes, has_more = find_complete_boxed_spans(text)

        self.assertEqual([box.content for box in boxes], [r"1+\frac{2}{3}", "4"])
        self.assertFalse(has_more)
        self.assertEqual(text[boxes[0].start : boxes[0].end], boxes[0].raw)

    def test_ignores_incomplete_box(self):
        boxes, has_more = find_complete_boxed_spans(r"x \boxed{1")

        self.assertEqual(boxes, [])
        self.assertFalse(has_more)

    def test_caps_at_first_ten_complete_boxes(self):
        text = " ".join(rf"\boxed{{{index}}}" for index in range(12))

        boxes, has_more = find_complete_boxed_spans(text, max_boxes=10)

        self.assertEqual(len(boxes), 10)
        self.assertEqual(boxes[-1].content, "9")
        self.assertTrue(has_more)

    def test_splits_steps_at_each_box_end_and_keeps_tail(self):
        text = r"first \boxed{1} second \boxed{2} tail"

        steps, boxes, has_more = split_response_at_boxed_ends(text)

        self.assertEqual(
            steps,
            [r"first \boxed{1}", r" second \boxed{2}", " tail"],
        )
        self.assertEqual(len(boxes), 2)
        self.assertFalse(has_more)


class FirstCorrectBoxAnalysisTest(unittest.TestCase):
    def test_first_correct_box_defines_external_redundancy(self):
        text = r"work \boxed{3} retry \boxed{5} repeat \boxed{5}"

        analysis = analyze_first_correct_box(
            text,
            is_correct_box=lambda box: box.content == "5",
            token_count=len,
            post_fcs_tolerance_tokens=0,
        )

        second_box_end = text.index(r"\boxed{5}") + len(r"\boxed{5}")
        expected_erd = round(len(text[second_box_end:]) / len(text), 2)
        self.assertEqual(analysis.first_correct_box_index, 2)
        self.assertEqual(analysis.fcs_char_end, second_box_end)
        self.assertEqual(analysis.external_redundancy, expected_erd)
        self.assertEqual(analysis.reward_factor, 1.0 - expected_erd)

    def test_no_correct_box_leaves_factor_one(self):
        analysis = analyze_first_correct_box(
            r"\boxed{1} \boxed{2}",
            is_correct_box=lambda box: False,
            token_count=len,
        )

        self.assertFalse(analysis.found)
        self.assertEqual(analysis.reward_factor, 1.0)

    def test_first_32_post_fcs_tokens_are_free(self):
        boxed = r"work \boxed{5}"
        within_tolerance = boxed + "x" * 32
        over_tolerance = boxed + "x" * 64

        within = analyze_first_correct_box(
            within_tolerance,
            is_correct_box=lambda box: box.content == "5",
            token_count=len,
            post_fcs_tolerance_tokens=32,
        )
        over = analyze_first_correct_box(
            over_tolerance,
            is_correct_box=lambda box: box.content == "5",
            token_count=len,
            post_fcs_tolerance_tokens=32,
        )

        self.assertEqual(within.external_redundancy, 0.0)
        self.assertEqual(within.reward_factor, 1.0)
        self.assertEqual(over.penalized_post_fcs_token_count, 32)
        self.assertEqual(
            over.external_redundancy,
            round(32 / len(over_tolerance), 2),
        )

    def test_negative_tolerance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            analyze_first_correct_box(
                r"\boxed{5}",
                is_correct_box=lambda box: True,
                token_count=len,
                post_fcs_tolerance_tokens=-1,
            )


class FirstCorrectBoxBatchRewardTest(unittest.TestCase):
    def _batch(self, response_texts, prefixes=None, correctness=None):
        tokenizer = CharacterTokenizer()
        encoded = [tokenizer.encode(text) for text in response_texts]
        width = max(len(ids) for ids in encoded)
        responses = torch.zeros((len(encoded), width), dtype=torch.long)
        mask = torch.zeros_like(responses)
        for row, ids in enumerate(encoded):
            responses[row, : len(ids)] = torch.tensor(ids)
            mask[row, : len(ids)] = 1
        return (
            tokenizer,
            responses,
            mask,
            prefixes or [0] * len(encoded),
            correctness or [1] * len(encoded),
        )

    @staticmethod
    def _score(*, solution_str, ground_truth, **kwargs):
        del kwargs
        return {
            "score": float(solution_str == rf"\boxed{{{ground_truth}}}"),
            "acc": solution_str == rf"\boxed{{{ground_truth}}}",
        }

    def test_repeated_correct_answer_receives_smaller_scalar_reward(self):
        short = r"work \boxed{5}"
        repeated = short + (r" again \boxed{5}" * 4)
        tokenizer, responses, mask, prefixes, correctness = self._batch(
            [short, repeated]
        )
        rewards = torch.zeros_like(responses, dtype=torch.float32)
        rewards[:, -1] = 1.0

        shaped, extras, metrics = apply_first_correct_box_reward(
            rewards,
            responses=responses,
            response_attention_mask=mask,
            partial_response_lengths=prefixes,
            correctness=correctness,
            tokenizer=tokenizer,
            compute_score=self._score,
            data_sources=["math", "math"],
            reward_models=[{"ground_truth": "5"}, {"ground_truth": "5"}],
        )

        self.assertAlmostEqual(float(shaped[0].sum()), 1.0)
        self.assertLess(float(shaped[1].sum()), 1.0)
        self.assertEqual(extras["first_correct_box_index"], [1, 1])
        self.assertEqual(metrics["n_found"], 2.0)
        self.assertEqual(metrics["n_penalized"], 1.0)

    def test_incorrect_final_rollout_is_never_shaped(self):
        text = r"work \boxed{5} then \boxed{6}"
        tokenizer, responses, mask, prefixes, _ = self._batch([text])
        rewards = torch.zeros_like(responses, dtype=torch.float32)
        rewards[0, -1] = -1.0

        shaped, extras, metrics = apply_first_correct_box_reward(
            rewards,
            responses=responses,
            response_attention_mask=mask,
            partial_response_lengths=prefixes,
            correctness=[0],
            tokenizer=tokenizer,
            compute_score=self._score,
            data_sources=["math"],
            reward_models=[{"ground_truth": "5"}],
        )

        torch.testing.assert_close(shaped, rewards)
        self.assertEqual(metrics["n_eligible_correct"], 0.0)
        self.assertEqual(extras["first_correct_box_found"], [False])

    def test_injected_prefix_is_excluded_from_length_ratio_and_box_scan(self):
        injected = r"noise \boxed{0} "
        generated = r"work \boxed{5} tail"
        full = injected + generated
        tokenizer, responses, mask, _, correctness = self._batch([full])
        rewards = torch.zeros_like(responses, dtype=torch.float32)
        rewards[0, -1] = 1.0

        _, extras, _ = apply_first_correct_box_reward(
            rewards,
            responses=responses,
            response_attention_mask=mask,
            partial_response_lengths=[len(injected)],
            correctness=correctness,
            tokenizer=tokenizer,
            compute_score=self._score,
            data_sources=["math"],
            reward_models=[{"ground_truth": "5"}],
        )

        self.assertEqual(extras["first_correct_box_index"], [1])
        self.assertEqual(
            extras["first_correct_box_total_token_count"], [len(generated)]
        )

    def test_only_first_ten_boxes_are_verified(self):
        text = " ".join(
            [rf"\boxed{{{index}}}" for index in range(10)] + [r"\boxed{99}"]
        )
        tokenizer, responses, mask, prefixes, correctness = self._batch([text])
        rewards = torch.zeros_like(responses, dtype=torch.float32)
        rewards[0, -1] = 1.0

        shaped, extras, metrics = apply_first_correct_box_reward(
            rewards,
            responses=responses,
            response_attention_mask=mask,
            partial_response_lengths=prefixes,
            correctness=correctness,
            tokenizer=tokenizer,
            compute_score=self._score,
            data_sources=["math"],
            reward_models=[{"ground_truth": "99"}],
            max_boxes=10,
        )

        torch.testing.assert_close(shaped, rewards)
        self.assertEqual(extras["first_correct_box_checked_count"], [10])
        self.assertEqual(extras["first_correct_box_has_more"], [True])
        self.assertEqual(metrics["n_found"], 0.0)


if __name__ == "__main__":
    unittest.main()
