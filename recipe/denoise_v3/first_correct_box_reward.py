"""First-correct-box external-redundancy reward for DenoiseRL v3.

This is the external-redundancy component of *Reconsidering Overthinking in
Reasoning Large Language Models* (HenryZhen97/Reconsidering-Overthinking),
adapted to mathematical rollouts with explicit ``\\boxed{...}`` answers.

For an already-correct rollout, the first correct box defines the First
Correct Solution (FCS).  Text after that box is external redundancy:

    ERD = round(max(post_fcs_tokens - tolerance_tokens, 0) / total_tokens, 2)
    reward = accuracy_reward * (1 - ERD)

The shaping remains sequence-level and does not modify GRPO advantage
calculation.  At most the first ``max_boxes`` complete boxes are verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class BoxedSpan:
    """Character span for one complete, balanced ``\\boxed{...}`` expression."""

    start: int
    end: int
    content: str
    raw: str


@dataclass(frozen=True)
class FirstCorrectBoxAnalysis:
    """External-redundancy measurements for one response."""

    candidate_box_count: int
    verified_box_count: int
    has_more_boxes: bool
    first_correct_box_index: int | None
    fcs_char_end: int | None
    fcs_token_count: int
    post_fcs_token_count: int
    penalized_post_fcs_token_count: int
    total_token_count: int
    post_fcs_tolerance_tokens: int
    external_redundancy: float
    reward_factor: float

    @property
    def found(self) -> bool:
        return self.first_correct_box_index is not None


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def find_complete_boxed_spans(
    text: str,
    *,
    max_boxes: int = 10,
) -> tuple[list[BoxedSpan], bool]:
    """Return the first complete balanced boxes and whether more were present.

    Whitespace between ``\\boxed`` and ``{`` is accepted.  Nested braces in the
    boxed answer are handled without a depth limit.  Incomplete boxes are
    ignored.
    """

    if max_boxes <= 0:
        raise ValueError(f"max_boxes must be > 0, got {max_boxes}.")

    spans: list[BoxedSpan] = []
    marker = "\\boxed"
    search_from = 0
    has_more = False

    while search_from < len(text):
        marker_start = text.find(marker, search_from)
        if marker_start < 0:
            break

        open_brace = marker_start + len(marker)
        while open_brace < len(text) and text[open_brace].isspace():
            open_brace += 1
        if open_brace >= len(text) or text[open_brace] != "{":
            search_from = marker_start + len(marker)
            continue

        depth = 0
        close_brace: int | None = None
        for index in range(open_brace, len(text)):
            char = text[index]
            if char == "{" and not _is_escaped(text, index):
                depth += 1
            elif char == "}" and not _is_escaped(text, index):
                depth -= 1
                if depth == 0:
                    close_brace = index
                    break

        if close_brace is None:
            # Continue searching in case a later independent box is complete.
            search_from = open_brace + 1
            continue

        if len(spans) >= max_boxes:
            has_more = True
            break

        end = close_brace + 1
        spans.append(
            BoxedSpan(
                start=marker_start,
                end=end,
                content=text[open_brace + 1 : close_brace],
                raw=text[marker_start:end],
            )
        )
        search_from = end

    return spans, has_more


def split_response_at_boxed_ends(
    text: str,
    *,
    max_boxes: int = 10,
) -> tuple[list[str], list[BoxedSpan], bool]:
    """Split a response after every retained boxed answer.

    The final element is the post-box tail when one exists.  With no boxes, the
    original non-empty response is returned as one step.
    """

    boxes, has_more = find_complete_boxed_spans(text, max_boxes=max_boxes)
    steps: list[str] = []
    step_start = 0
    for box in boxes:
        steps.append(text[step_start : box.end])
        step_start = box.end
    if step_start < len(text):
        steps.append(text[step_start:])
    return steps, boxes, has_more


def analyze_first_correct_box(
    response_text: str,
    *,
    is_correct_box: Callable[[BoxedSpan], bool],
    token_count: Callable[[str], int],
    max_boxes: int = 10,
    erd_round_digits: int = 2,
    post_fcs_tolerance_tokens: int = 32,
) -> FirstCorrectBoxAnalysis:
    """Find the first correct box and calculate the paper's ERD multiplier."""

    if erd_round_digits < 0:
        raise ValueError(
            f"erd_round_digits must be non-negative, got {erd_round_digits}."
        )
    if post_fcs_tolerance_tokens < 0:
        raise ValueError(
            "post_fcs_tolerance_tokens must be non-negative, got "
            f"{post_fcs_tolerance_tokens}."
        )

    boxes, has_more = find_complete_boxed_spans(
        response_text, max_boxes=max_boxes
    )
    total_tokens = max(0, int(token_count(response_text)))

    for box_index, box in enumerate(boxes, start=1):
        if not is_correct_box(box):
            continue

        fcs_tokens = max(0, int(token_count(response_text[: box.end])))
        post_fcs_tokens = max(0, int(token_count(response_text[box.end :])))
        penalized_post_fcs_tokens = max(
            post_fcs_tokens - post_fcs_tolerance_tokens, 0
        )
        if total_tokens == 0:
            external_redundancy = 0.0
        else:
            external_redundancy = round(
                penalized_post_fcs_tokens / total_tokens, erd_round_digits
            )
            external_redundancy = min(1.0, max(0.0, external_redundancy))

        return FirstCorrectBoxAnalysis(
            candidate_box_count=len(boxes),
            verified_box_count=box_index,
            has_more_boxes=has_more,
            first_correct_box_index=box_index,
            fcs_char_end=box.end,
            fcs_token_count=fcs_tokens,
            post_fcs_token_count=post_fcs_tokens,
            penalized_post_fcs_token_count=penalized_post_fcs_tokens,
            total_token_count=total_tokens,
            post_fcs_tolerance_tokens=post_fcs_tolerance_tokens,
            external_redundancy=external_redundancy,
            reward_factor=1.0 - external_redundancy,
        )

    return FirstCorrectBoxAnalysis(
        candidate_box_count=len(boxes),
        verified_box_count=len(boxes),
        has_more_boxes=has_more,
        first_correct_box_index=None,
        fcs_char_end=None,
        fcs_token_count=0,
        post_fcs_token_count=0,
        penalized_post_fcs_token_count=0,
        total_token_count=total_tokens,
        post_fcs_tolerance_tokens=post_fcs_tolerance_tokens,
        external_redundancy=0.0,
        reward_factor=1.0,
    )


def _decode(tokenizer: Any, token_ids: torch.Tensor) -> str:
    ids = token_ids.detach().cpu().tolist()
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(ids, skip_special_tokens=True)


def _token_count(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return len(token_ids)


def _score_is_correct(score: Any) -> bool:
    if isinstance(score, Mapping):
        value = score.get("acc", score.get("score", 0.0))
    else:
        value = score
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            "Box verifier must return a scalar 'acc'/'score', "
            f"got shape {array.shape}."
        )
    return float(array.reshape(-1)[0]) > 0.0


def apply_first_correct_box_reward(
    reward_tensor: torch.Tensor,
    *,
    responses: torch.Tensor,
    response_attention_mask: torch.Tensor,
    partial_response_lengths: Sequence[int],
    correctness: Sequence[Any],
    tokenizer: Any,
    compute_score: Callable[..., Any],
    data_sources: Sequence[Any],
    reward_models: Sequence[Mapping[str, Any]],
    extra_infos: Sequence[Any] | None = None,
    max_boxes: int = 10,
    erd_round_digits: int = 2,
    post_fcs_tolerance_tokens: int = 32,
) -> tuple[torch.Tensor, dict[str, list[Any]], dict[str, float]]:
    """Apply FCA external-redundancy shaping to a rollout batch.

    Only rows whose original final answer is correct are eligible.  Boxes are
    verified independently with the active reward manager's ``compute_score``.
    Injected DenoiseRL prefixes are excluded from all length measurements.
    """

    if reward_tensor.ndim != 2:
        raise ValueError(
            f"reward_tensor must be 2D, got shape {tuple(reward_tensor.shape)}."
        )
    if responses.ndim != 2 or response_attention_mask.shape != responses.shape:
        raise ValueError(
            "responses and response_attention_mask must be equally shaped 2D tensors."
        )
    if responses.shape != reward_tensor.shape:
        raise ValueError(
            "responses and reward_tensor must have the same shape, got "
            f"{tuple(responses.shape)} and {tuple(reward_tensor.shape)}."
        )
    if post_fcs_tolerance_tokens < 0:
        raise ValueError(
            "post_fcs_tolerance_tokens must be non-negative, got "
            f"{post_fcs_tolerance_tokens}."
        )

    batch_size = reward_tensor.shape[0]
    fields = {
        "partial_response_lengths": partial_response_lengths,
        "correctness": correctness,
        "data_sources": data_sources,
        "reward_models": reward_models,
    }
    if extra_infos is not None:
        fields["extra_infos"] = extra_infos
    for name, values in fields.items():
        if len(values) != batch_size:
            raise ValueError(
                f"{name} must have {batch_size} values, got {len(values)}."
            )

    shaped_reward = reward_tensor.clone()
    reward_before = reward_tensor.sum(dim=-1)
    analyses: list[FirstCorrectBoxAnalysis | None] = [None] * batch_size
    eligible = np.asarray(correctness).reshape(-1).astype(np.float64) > 0.0

    for row_index in range(batch_size):
        if not eligible[row_index]:
            continue

        valid_length = int(response_attention_mask[row_index].sum().item())
        prefix_length = int(partial_response_lengths[row_index])
        if prefix_length < 0 or prefix_length > valid_length:
            raise ValueError(
                "partial_response_lengths must be within each valid response: "
                f"row {row_index} has prefix {prefix_length}, valid length {valid_length}."
            )

        generated_ids = responses[row_index, prefix_length:valid_length]
        generated_text = _decode(tokenizer, generated_ids)
        reward_model = reward_models[row_index]
        if not isinstance(reward_model, Mapping) or "ground_truth" not in reward_model:
            raise ValueError(
                f"reward_models[{row_index}] must contain 'ground_truth'."
            )
        extra_info = (
            extra_infos[row_index] if extra_infos is not None else {}
        )
        if isinstance(extra_info, Mapping):
            extra_info = dict(extra_info)

        def verify_box(box: BoxedSpan) -> bool:
            score = compute_score(
                data_source=data_sources[row_index],
                solution_str=f"\\boxed{{{box.content}}}",
                ground_truth=reward_model["ground_truth"],
                extra_info=extra_info,
            )
            return _score_is_correct(score)

        analysis = analyze_first_correct_box(
            generated_text,
            is_correct_box=verify_box,
            token_count=lambda text: _token_count(tokenizer, text),
            max_boxes=max_boxes,
            erd_round_digits=erd_round_digits,
            post_fcs_tolerance_tokens=post_fcs_tolerance_tokens,
        )
        analyses[row_index] = analysis
        if analysis.found:
            shaped_reward[row_index] *= analysis.reward_factor

    reward_after = shaped_reward.sum(dim=-1)
    effective_penalties = reward_before - reward_after

    extras: dict[str, list[Any]] = {
        "first_correct_box_found": [],
        "first_correct_box_candidate_count": [],
        "first_correct_box_checked_count": [],
        "first_correct_box_has_more": [],
        "first_correct_box_index": [],
        "first_correct_box_fcs_token_count": [],
        "first_correct_box_post_fcs_token_count": [],
        "first_correct_box_penalized_post_fcs_token_count": [],
        "first_correct_box_total_token_count": [],
        "first_correct_box_post_fcs_tolerance_tokens": [],
        "external_redundancy": [],
        "external_redundancy_factor": [],
        "external_redundancy_penalty": effective_penalties.detach()
        .cpu()
        .tolist(),
    }
    for analysis in analyses:
        extras["first_correct_box_found"].append(
            bool(analysis is not None and analysis.found)
        )
        extras["first_correct_box_candidate_count"].append(
            analysis.candidate_box_count if analysis is not None else 0
        )
        extras["first_correct_box_checked_count"].append(
            analysis.verified_box_count if analysis is not None else 0
        )
        extras["first_correct_box_has_more"].append(
            bool(analysis is not None and analysis.has_more_boxes)
        )
        extras["first_correct_box_index"].append(
            analysis.first_correct_box_index
            if analysis is not None and analysis.found
            else -1
        )
        extras["first_correct_box_fcs_token_count"].append(
            analysis.fcs_token_count if analysis is not None else 0
        )
        extras["first_correct_box_post_fcs_token_count"].append(
            analysis.post_fcs_token_count if analysis is not None else 0
        )
        extras["first_correct_box_penalized_post_fcs_token_count"].append(
            analysis.penalized_post_fcs_token_count
            if analysis is not None
            else 0
        )
        extras["first_correct_box_total_token_count"].append(
            analysis.total_token_count if analysis is not None else 0
        )
        extras["first_correct_box_post_fcs_tolerance_tokens"].append(
            analysis.post_fcs_tolerance_tokens
            if analysis is not None
            else post_fcs_tolerance_tokens
        )
        extras["external_redundancy"].append(
            analysis.external_redundancy if analysis is not None else 0.0
        )
        extras["external_redundancy_factor"].append(
            analysis.reward_factor if analysis is not None else 1.0
        )

    found_analyses = [
        analysis for analysis in analyses if analysis is not None and analysis.found
    ]
    eligible_analyses = [analysis for analysis in analyses if analysis is not None]
    eligible_mask = torch.as_tensor(
        eligible, device=reward_tensor.device, dtype=torch.bool
    )
    metrics = {
        "n_eligible_correct": float(len(eligible_analyses)),
        "n_found": float(len(found_analyses)),
        "found_ratio": (
            float(len(found_analyses)) / float(len(eligible_analyses))
            if eligible_analyses
            else 0.0
        ),
        "n_penalized": float(
            sum(analysis.reward_factor < 1.0 for analysis in found_analyses)
        ),
        "n_box_limit_reached": float(
            sum(analysis.has_more_boxes for analysis in eligible_analyses)
        ),
        "checked_box_count_mean": (
            float(np.mean([a.verified_box_count for a in eligible_analyses]))
            if eligible_analyses
            else 0.0
        ),
        "candidate_box_count_mean": (
            float(np.mean([a.candidate_box_count for a in eligible_analyses]))
            if eligible_analyses
            else 0.0
        ),
        "first_correct_box_index_mean": (
            float(np.mean([a.first_correct_box_index for a in found_analyses]))
            if found_analyses
            else 0.0
        ),
        "external_redundancy_mean": (
            float(np.mean([a.external_redundancy for a in found_analyses]))
            if found_analyses
            else 0.0
        ),
        "factor_mean": (
            float(np.mean([a.reward_factor for a in found_analyses]))
            if found_analyses
            else 1.0
        ),
        "post_fcs_token_count_mean": (
            float(np.mean([a.post_fcs_token_count for a in found_analyses]))
            if found_analyses
            else 0.0
        ),
        "penalized_post_fcs_token_count_mean": (
            float(
                np.mean(
                    [a.penalized_post_fcs_token_count for a in found_analyses]
                )
            )
            if found_analyses
            else 0.0
        ),
        "post_fcs_tolerance_tokens": float(post_fcs_tolerance_tokens),
        "reward_before_mean": (
            float(reward_before[eligible_mask].mean().item())
            if np.any(eligible)
            else 0.0
        ),
        "reward_after_mean": (
            float(reward_after[eligible_mask].mean().item())
            if np.any(eligible)
            else 0.0
        ),
    }
    return shaped_reward, extras, metrics
