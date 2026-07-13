"""Helpers for composing loss aggregates across micro-batches.

``agg_loss`` first reduces a loss matrix inside one micro-batch. To recover
the same objective after splitting a mini-batch, each local aggregate must be
weighted by the denominator used by that aggregation mode. Weighting by the
number of rows is only correct for sequence-mean objectives; it is incorrect
for token-mean when rows have different numbers of trainable tokens.
"""


_SEQUENCE_MEAN_MODES = {
    "seq-mean-token-sum",
    "seq-mean-token-mean",
}


def aggregation_mass(token_count: float, active_sequence_count: float, loss_agg_mode: str) -> float:
    """Return the additive denominator mass used by ``agg_loss``."""

    if loss_agg_mode == "token-mean":
        return float(token_count)
    if loss_agg_mode in _SEQUENCE_MEAN_MODES or loss_agg_mode.startswith(
        "seq-mean-token-length-norm-"
    ):
        return float(active_sequence_count)
    if loss_agg_mode == "seq-mean-token-sum-norm":
        # This mode is a sum divided by the fixed response width, not a mean
        # over rows/tokens. Micro-batch contributions should simply be added.
        return 1.0
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def micro_batch_aggregation_scale(
    *,
    local_token_count: float,
    local_active_sequence_count: float,
    global_mass: float,
    loss_agg_mode: str,
) -> float:
    """Scale a local ``agg_loss`` value into its mini-batch contribution."""

    if local_token_count <= 0 or local_active_sequence_count <= 0:
        return 0.0
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return 1.0
    local_mass = aggregation_mass(
        token_count=local_token_count,
        active_sequence_count=local_active_sequence_count,
        loss_agg_mode=loss_agg_mode,
    )
    if global_mass <= 0:
        return 0.0
    return local_mass / float(global_mass)
