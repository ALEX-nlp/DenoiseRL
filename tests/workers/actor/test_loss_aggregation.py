import importlib.util
import unittest
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "verl"
    / "workers"
    / "actor"
    / "loss_aggregation.py"
)
_SPEC = importlib.util.spec_from_file_location("loss_aggregation_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


class TestMicroBatchAggregationScale(unittest.TestCase):
    def test_token_mean_uses_trainable_token_count(self):
        long_rows = 2
        short_rows = 16
        long_tokens = 2_000
        short_tokens = 16
        global_tokens = long_tokens + short_tokens

        long_scale = _MODULE.micro_batch_aggregation_scale(
            local_token_count=long_tokens,
            local_active_sequence_count=long_rows,
            global_mass=global_tokens,
            loss_agg_mode="token-mean",
        )
        short_scale = _MODULE.micro_batch_aggregation_scale(
            local_token_count=short_tokens,
            local_active_sequence_count=short_rows,
            global_mass=global_tokens,
            loss_agg_mode="token-mean",
        )

        long_loss = 1.0
        short_loss = 2.0
        composed = long_scale * long_loss + short_scale * short_loss
        direct = (long_tokens * long_loss + short_tokens * short_loss) / global_tokens
        self.assertAlmostEqual(composed, direct)

        old_row_weighted = (
            long_rows / (long_rows + short_rows) * long_loss
            + short_rows / (long_rows + short_rows) * short_loss
        )
        self.assertGreater(old_row_weighted - long_loss, 100 * (direct - long_loss))

    def test_sequence_mean_uses_active_sequence_count(self):
        global_sequences = 18
        long_scale = _MODULE.micro_batch_aggregation_scale(
            local_token_count=2_000,
            local_active_sequence_count=2,
            global_mass=global_sequences,
            loss_agg_mode="seq-mean-token-mean",
        )
        short_scale = _MODULE.micro_batch_aggregation_scale(
            local_token_count=16,
            local_active_sequence_count=16,
            global_mass=global_sequences,
            loss_agg_mode="seq-mean-token-mean",
        )
        self.assertAlmostEqual(long_scale, 2 / 18)
        self.assertAlmostEqual(short_scale, 16 / 18)

    def test_empty_micro_batch_contributes_zero(self):
        scale = _MODULE.micro_batch_aggregation_scale(
            local_token_count=0,
            local_active_sequence_count=0,
            global_mass=100,
            loss_agg_mode="token-mean",
        )
        self.assertEqual(scale, 0.0)

    def test_sum_norm_micro_batches_are_additive(self):
        scale = _MODULE.micro_batch_aggregation_scale(
            local_token_count=20,
            local_active_sequence_count=2,
            global_mass=1,
            loss_agg_mode="seq-mean-token-sum-norm",
        )
        self.assertEqual(scale, 1.0)


if __name__ == "__main__":
    unittest.main()
