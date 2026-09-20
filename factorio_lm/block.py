"""Stateful exact emulator for one FactorioLM Transformer block."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .attention import CONTEXT_DEPTH, MODEL_WIDTH, AttentionTrace, attention_bank, deterministic_relative_biases
from .emulator import (
    KVStorage,
    integer_linear,
    maxnorm,
    rational_scale,
    relu,
    residual_add,
    validate_integer_matrix,
)


def deterministic_ternary_matrix(rows: int, columns: int, seed: int) -> list[list[int]]:
    """Reproducible ~50% sparse matrix with both ternary signs in every stage."""

    if rows <= 0 or columns <= 0:
        raise ValueError("matrix dimensions must be positive")
    matrix: list[list[int]] = []
    for row in range(rows):
        values: list[int] = []
        for column in range(columns):
            code = (row * 37 + column * 19 + seed * 13 + row * column * 3) % 8
            values.append(0 if code < 4 else (1 if code < 6 else -1))
        matrix.append(values)
    return matrix


@dataclass(frozen=True)
class BlockWeights:
    query: list[list[int]]
    key: list[list[int]]
    value: list[list[int]]
    output: list[list[int]]
    ffn_up: list[list[int]]
    ffn_down: list[list[int]]

    @classmethod
    def deterministic(cls) -> "BlockWeights":
        return cls(
            deterministic_ternary_matrix(32, 32, 1),
            deterministic_ternary_matrix(32, 32, 2),
            deterministic_ternary_matrix(32, 32, 3),
            deterministic_ternary_matrix(32, 32, 4),
            deterministic_ternary_matrix(64, 32, 5),
            deterministic_ternary_matrix(32, 64, 6),
        )

    def validate(self, maximum_absolute_weight: int = 1) -> None:
        shapes = ((self.query, 32, 32), (self.key, 32, 32), (self.value, 32, 32),
                  (self.output, 32, 32), (self.ffn_up, 64, 32), (self.ffn_down, 32, 64))
        for matrix, rows, columns in shapes:
            actual_rows, actual_columns = validate_integer_matrix(
                matrix, maximum_absolute_weight=maximum_absolute_weight,
            )
            if (actual_rows, actual_columns) != (rows, columns):
                raise ValueError(f"matrix must have shape {rows}x{columns}")

    def sparsity_report(self) -> dict[str, dict[str, int | float]]:
        result: dict[str, dict[str, int | float]] = {}
        for name in ("query", "key", "value", "output", "ffn_up", "ffn_down"):
            matrix = getattr(self, name)
            total = len(matrix) * len(matrix[0])
            nonzero = sum(weight != 0 for row in matrix for weight in row)
            result[name] = {
                "total": total,
                "nonzero": nonzero,
                "zero": total - nonzero,
                "zero_percentage": 100.0 * (total - nonzero) / total,
            }
        return result


@dataclass(frozen=True)
class MatrixScale:
    numerator: int = 1
    denominator: int = 1

    def validate(self) -> None:
        if isinstance(self.numerator, bool) or not isinstance(self.numerator, int):
            raise ValueError("matrix scale numerator must be an integer")
        if isinstance(self.denominator, bool) or not isinstance(self.denominator, int):
            raise ValueError("matrix scale denominator must be an integer")
        if self.denominator <= 0:
            raise ValueError("matrix scale denominator must be positive")
        if self.numerator <= 0:
            raise ValueError("matrix scale numerator must be positive")

    def apply(self, vector: Sequence[int]) -> list[int]:
        self.validate()
        return rational_scale(vector, self.numerator, self.denominator)

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return {"numerator": self.numerator, "denominator": self.denominator}


@dataclass(frozen=True)
class BlockScales:
    query: MatrixScale = MatrixScale()
    key: MatrixScale = MatrixScale()
    value: MatrixScale = MatrixScale()
    output: MatrixScale = MatrixScale()
    ffn_up: MatrixScale = MatrixScale()
    ffn_down: MatrixScale = MatrixScale()

    def validate(self) -> None:
        for name in ("query", "key", "value", "output", "ffn_up", "ffn_down"):
            getattr(self, name).validate()

    def to_dict(self) -> dict[str, dict[str, int]]:
        self.validate()
        return {name: getattr(self, name).to_dict()
                for name in ("query", "key", "value", "output", "ffn_up", "ffn_down")}


@dataclass(frozen=True)
class BlockTrace:
    step: int
    block_input: list[int]
    norm1: list[int]
    query: list[int]
    key: list[int]
    value: list[int]
    cache_keys: list[list[int]]
    cache_values: list[list[int]]
    cache_validity: list[bool]
    attention: AttentionTrace
    output_projection: list[int]
    residual1: list[int]
    norm2: list[int]
    ffn_up: list[int]
    ffn_relu: list[int]
    ffn_down: list[int]
    block_output: list[int]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["attention"] = self.attention.to_dict()
        return value


class TransformerBlockEmulator:
    """One pre-norm block with causal current-token commit-before-attention."""

    def __init__(
        self,
        *,
        weights: BlockWeights | None = None,
        relative_biases: Sequence[Sequence[int]] | None = None,
        score_divisor: int = 32,
        depth: int = CONTEXT_DEPTH,
        scales: BlockScales | None = None,
        maximum_absolute_weight: int = 1,
    ) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        if score_divisor == 0:
            raise ValueError("score divisor cannot be zero")
        self.weights = weights or BlockWeights.deterministic()
        self.weights.validate(maximum_absolute_weight)
        self.scales = scales or BlockScales()
        self.scales.validate()
        self.maximum_absolute_weight = maximum_absolute_weight
        self.relative_biases = [list(row) for row in (relative_biases or deterministic_relative_biases(depth))]
        if len(self.relative_biases) != 4 or any(len(row) != depth for row in self.relative_biases):
            raise ValueError(f"relative biases must have shape 4x{depth}")
        self.score_divisor = score_divisor
        self.cache = KVStorage(depth=depth, width=MODEL_WIDTH)
        self.steps = 0

    def reset(self) -> None:
        self.cache.step(reset=True)
        self.steps = 0

    def step(self, vector: Sequence[int]) -> BlockTrace:
        if len(vector) != MODEL_WIDTH:
            raise ValueError(f"block input must have width {MODEL_WIDTH}")
        block_input = list(vector)
        norm1 = maxnorm(block_input)
        query = self.scales.query.apply(integer_linear(
            self.weights.query, norm1, maximum_absolute_weight=self.maximum_absolute_weight))
        key = self.scales.key.apply(integer_linear(
            self.weights.key, norm1, maximum_absolute_weight=self.maximum_absolute_weight))
        value = self.scales.value.apply(integer_linear(
            self.weights.value, norm1, maximum_absolute_weight=self.maximum_absolute_weight))

        # This is the causal boundary: K_t/V_t become physical age zero before
        # Q_t observes the attention bank, so the current token attends to self.
        cache = self.cache.step(key, value, commit=True)
        attention = attention_bank(
            query,
            cache["keys"],
            cache["values"],
            cache["validity"],
            relative_biases=self.relative_biases,
            score_divisor=self.score_divisor,
        )
        output_projection = self.scales.output.apply(integer_linear(
            self.weights.output, attention.output,
            maximum_absolute_weight=self.maximum_absolute_weight))
        residual1 = residual_add(block_input, output_projection)
        norm2 = maxnorm(residual1)
        ffn_up = self.scales.ffn_up.apply(integer_linear(
            self.weights.ffn_up, norm2, maximum_absolute_weight=self.maximum_absolute_weight))
        ffn_relu = relu(ffn_up)
        ffn_down = self.scales.ffn_down.apply(integer_linear(
            self.weights.ffn_down, ffn_relu,
            maximum_absolute_weight=self.maximum_absolute_weight))
        block_output = residual_add(residual1, ffn_down)
        trace = BlockTrace(
            self.steps,
            block_input,
            norm1,
            query,
            key,
            value,
            cache["keys"],
            cache["values"],
            cache["validity"],
            attention,
            output_projection,
            residual1,
            norm2,
            ffn_up,
            ffn_relu,
            ffn_down,
            block_output,
        )
        self.steps += 1
        return trace


def deterministic_token_inputs(count: int = 3) -> list[list[int]]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [
        [((token + 3) * (channel * 7 + 5) % 47) - 23 for channel in range(MODEL_WIDTH)]
        for token in range(count)
    ]
