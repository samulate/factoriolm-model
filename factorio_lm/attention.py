"""Exact Run 2 causal attention emulation.

The implementation spells out every wrapped integer operation rather than
using a floating-point reference model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .int32 import add, clamp, left_shift, mul, trunc_div


HEADS = 4
HEAD_WIDTH = 8
MODEL_WIDTH = HEADS * HEAD_WIDTH
CONTEXT_DEPTH = 64
SCORE_MIN = -6
SCORE_MAX = 6


def _vector(values: Sequence[int], width: int, name: str) -> list[int]:
    if len(values) != width:
        raise ValueError(f"{name} must have width {width}, got {len(values)}")
    # add(x, 0) performs the same int32 coercion as a Factorio network.
    return [add(value, 0) for value in values]


def _biases(values: Sequence[Sequence[int]], depth: int) -> list[list[int]]:
    if len(values) != HEADS or any(len(row) != depth for row in values):
        raise ValueError(f"relative biases must have shape {HEADS}x{depth}")
    return [[add(value, 0) for value in row] for row in values]


@dataclass(frozen=True)
class AttentionCellTrace:
    valid: bool
    dots: list[int]
    scaled_scores: list[int]
    biased_scores: list[int]
    scores: list[int]
    raw_weights: list[int]
    weights: list[int]
    denominator_contributions: list[int]
    weighted_values: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionTrace:
    cells: list[AttentionCellTrace]
    denominators: list[int]
    numerators: list[int]
    output: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "denominators": list(self.denominators),
            "numerators": list(self.numerators),
            "output": list(self.output),
        }


def attention_cell(
    query: Sequence[int],
    key: Sequence[int],
    value: Sequence[int],
    *,
    valid: bool,
    relative_bias: Sequence[int],
    score_divisor: int,
) -> AttentionCellTrace:
    """Evaluate one physical age slot for all four heads."""

    q = _vector(query, MODEL_WIDTH, "query")
    k = _vector(key, MODEL_WIDTH, "key")
    v = _vector(value, MODEL_WIDTH, "value")
    bias = _vector(relative_bias, HEADS, "relative bias")
    if score_divisor == 0:
        raise ValueError("score divisor cannot be zero")

    dots: list[int] = []
    scaled: list[int] = []
    biased: list[int] = []
    scores: list[int] = []
    raw_weights: list[int] = []
    weights: list[int] = []
    weighted_values = [0] * MODEL_WIDTH
    for head in range(HEADS):
        start = head * HEAD_WIDTH
        dot = 0
        for channel in range(start, start + HEAD_WIDTH):
            dot = add(dot, mul(q[channel], k[channel]))
        scale = trunc_div(dot, score_divisor)
        biased_score = add(scale, bias[head])
        score = clamp(biased_score, SCORE_MIN, SCORE_MAX)
        raw_weight = left_shift(1, score + 6)
        weight = raw_weight if valid else 0
        dots.append(dot)
        scaled.append(scale)
        biased.append(biased_score)
        scores.append(score)
        raw_weights.append(raw_weight)
        weights.append(weight)
        for channel in range(start, start + HEAD_WIDTH):
            weighted_values[channel] = mul(weight, v[channel]) if valid else 0
    return AttentionCellTrace(
        valid=bool(valid),
        dots=dots,
        scaled_scores=scaled,
        biased_scores=biased,
        scores=scores,
        raw_weights=raw_weights,
        weights=weights,
        denominator_contributions=list(weights),
        weighted_values=weighted_values,
    )


def attention_bank(
    query: Sequence[int],
    keys: Sequence[Sequence[int]],
    values: Sequence[Sequence[int]],
    validity: Sequence[bool],
    *,
    relative_biases: Sequence[Sequence[int]],
    score_divisor: int,
) -> AttentionTrace:
    """Evaluate a spatial bank; list index is physical age (0 is newest)."""

    depth = len(keys)
    if depth <= 0:
        raise ValueError("attention bank needs at least one slot")
    if len(values) != depth or len(validity) != depth:
        raise ValueError("keys, values, and validity must have the same depth")
    biases = _biases(relative_biases, depth)
    cells = [
        attention_cell(
            query,
            keys[age],
            values[age],
            valid=bool(validity[age]),
            relative_bias=[biases[head][age] for head in range(HEADS)],
            score_divisor=score_divisor,
        )
        for age in range(depth)
    ]
    denominators = [0] * HEADS
    numerators = [0] * MODEL_WIDTH
    for cell in cells:
        for head, contribution in enumerate(cell.denominator_contributions):
            denominators[head] = add(denominators[head], contribution)
        for channel, contribution in enumerate(cell.weighted_values):
            numerators[channel] = add(numerators[channel], contribution)
    output = [
        trunc_div(numerators[channel], denominators[channel // HEAD_WIDTH])
        for channel in range(MODEL_WIDTH)
    ]
    return AttentionTrace(cells, denominators, numerators, output)


def deterministic_relative_biases(depth: int = CONTEXT_DEPTH) -> list[list[int]]:
    """Small reproducible 4xdepth fixture spanning useful negative/positive cases."""

    if depth <= 0:
        raise ValueError("depth must be positive")
    return [
        [((head * 5 + age * 3 + 1) % 9) - 4 for age in range(depth)]
        for head in range(HEADS)
    ]

