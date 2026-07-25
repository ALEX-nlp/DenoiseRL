import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import torch


def _load_compute_data_metrics():
    """Load metric_utils without importing the optional Ray training runtime."""
    verl_stub = types.ModuleType("verl")
    verl_stub.DataProto = object
    import_utils_stub = types.ModuleType("verl.utils.import_utils")
    import_utils_stub.deprecated = lambda _message: lambda function: function
    module_path = (
        Path(__file__).resolve().parents[2]
        / "verl"
        / "trainer"
        / "ppo"
        / "metric_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_denoise_v2_test_metric_utils",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "verl": verl_stub,
            "verl.utils": types.ModuleType("verl.utils"),
            "verl.utils.import_utils": import_utils_stub,
        },
    ):
        spec.loader.exec_module(module)
    return module.compute_data_metrics


compute_data_metrics = _load_compute_data_metrics()


class ResponseClipRatioMetricsTest(unittest.TestCase):
    @staticmethod
    def _batch():
        response_mask = torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.cat(
            [torch.ones(2, 2, dtype=torch.long), response_mask],
            dim=-1,
        )
        return SimpleNamespace(
            batch={
                "responses": torch.zeros(2, 3, dtype=torch.long),
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "token_level_scores": torch.zeros(2, 3),
                "token_level_rewards": torch.zeros(2, 3),
                "advantages": torch.zeros(2, 3),
                "returns": torch.zeros(2, 3),
            },
            non_tensor_batch={},
        )

    def test_default_includes_response_clip_ratios(self):
        metrics = compute_data_metrics(self._batch(), use_critic=False)

        self.assertEqual(metrics["response_length/clip_ratio"], 0.5)
        self.assertEqual(metrics["response_length_non_aborted/clip_ratio"], 0.5)

    def test_none_mode_can_use_continuation_lengths_for_clip_ratios(self):
        metrics = compute_data_metrics(
            self._batch(),
            use_critic=False,
            response_clip_lengths=torch.tensor([3, 3]),
            response_clip_max_length=3,
        )

        self.assertEqual(metrics["response_length/clip_ratio"], 1.0)
        self.assertEqual(metrics["response_length_non_aborted/clip_ratio"], 1.0)
        self.assertEqual(metrics["response_length/mean"], 2.5)
        self.assertEqual(metrics["response_length_non_aborted/mean"], 2.5)


if __name__ == "__main__":
    unittest.main()
