"""Utilities for cutting noisy solution prefixes at stable text boundaries."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Protocol


class _Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int], **kwargs) -> str: ...


@dataclass(frozen=True)
class PrefixCut:
    """Result of cutting a wrong solution into a noisy prefix."""

    text: str
    token_count: int
    total_token_count: int
    used_line_boundary: bool

    @property
    def realized_ratio(self) -> float:
        if self.total_token_count <= 0:
            return 0.0
        return self.token_count / self.total_token_count


def _token_cut(tokenizer: _Tokenizer, text: str, token_ids: list[int], ratio: float) -> PrefixCut:
    cut = int(len(token_ids) * ratio)
    cut = max(1, min(cut, len(token_ids)))
    prefix_text = tokenizer.decode(token_ids[:cut], skip_special_tokens=True)
    return PrefixCut(
        text=prefix_text,
        token_count=cut,
        total_token_count=len(token_ids),
        used_line_boundary=False,
    )


def _decoded_prefix_char_position(
    tokenizer: _Tokenizer,
    text: str,
    token_ids: list[int],
    target_token_count: float,
) -> int:
    """Map the target token position back to a nearby character position."""
    cut = max(1, min(int(target_token_count), len(token_ids)))
    try:
        decoded = tokenizer.decode(
            token_ids[:cut],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded = tokenizer.decode(token_ids[:cut], skip_special_tokens=True)

    if text.startswith(decoded):
        return len(decoded)

    # This position only locates the two neighboring line boundaries. Some slow
    # tokenizers normalize whitespace during decode, so use a proportional
    # character position when an exact prefix match is unavailable.
    return round(len(text) * target_token_count / len(token_ids))


def cut_wrong_solution_prefix(
    tokenizer: _Tokenizer,
    text: str,
    ratio: float,
    strategy: str = "token",
) -> PrefixCut:
    """Cut ``text`` by token ratio or round that ratio to the nearest line end.

    ``strategy="line"`` first locates the requested token position, then compares
    the complete line ending immediately before and after it. The candidate whose
    re-tokenized length is closest to ``ratio * total_tokens`` is selected. Ties go
    to the shorter prefix. Text without a useful line boundary falls back to the
    legacy token cut so a one-line solution is not expanded to its full length.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("Wrong solution must be a non-empty string.")
    if not (0.0 < float(ratio) <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}.")

    strategy = str(strategy).strip().lower()
    if strategy not in ("token", "line"):
        raise ValueError(
            "partial wrong cut strategy must be 'token' or 'line', "
            f"got {strategy!r}."
        )

    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not token_ids:
        raise ValueError("Empty wrong solution after tokenization.")
    if strategy == "token" or ratio == 1.0:
        return _token_cut(tokenizer, text, token_ids, ratio)

    lines = text.splitlines(keepends=True)
    # A single physical line has no meaningful interior line boundary. This also
    # handles a single line ending with a newline: its only boundary is full text.
    if len(lines) < 2:
        return _token_cut(tokenizer, text, token_ids, ratio)

    line_ends = []
    char_count = 0
    for line in lines:
        char_count += len(line)
        line_ends.append(char_count)
    if line_ends[-1] != len(text):  # defensive: splitlines should cover all text
        line_ends.append(len(text))

    target_token_count = len(token_ids) * float(ratio)
    target_char_position = _decoded_prefix_char_position(
        tokenizer, text, token_ids, target_token_count
    )
    right = bisect_left(line_ends, target_char_position)

    candidate_indexes = {min(right, len(line_ends) - 1)}
    if right > 0:
        candidate_indexes.add(right - 1)

    candidates = []
    for index in sorted(candidate_indexes):
        prefix_text = text[: line_ends[index]]
        prefix_token_count = len(tokenizer.encode(prefix_text, add_special_tokens=False))
        if prefix_token_count > 0:
            candidates.append((prefix_text, prefix_token_count))

    if not candidates:
        return _token_cut(tokenizer, text, token_ids, ratio)

    prefix_text, prefix_token_count = min(
        candidates,
        key=lambda item: (abs(item[1] - target_token_count), item[1]),
    )
    return PrefixCut(
        text=prefix_text,
        token_count=prefix_token_count,
        total_token_count=len(token_ids),
        used_line_boundary=True,
    )
