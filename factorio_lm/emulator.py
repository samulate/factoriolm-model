"""Exact deployment-oriented emulators for Run 1 primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .int32 import add, mul, neg, trunc_div, wrap


Vector = list[int]
Matrix = list[list[int]]


def _wrapped_vector(values: Iterable[int], *, width: int | None = None) -> Vector:
    result = [wrap(value) for value in values]
    if width is not None and len(result) != width:
        raise ValueError(f"expected vector width {width}, got {len(result)}")
    return result


def validate_ternary_matrix(matrix: Sequence[Sequence[int]]) -> tuple[int, int]:
    rows, width = validate_integer_matrix(matrix)
    for row_index, row in enumerate(matrix):
        invalid = [weight for weight in row if weight not in {-1, 0, 1}]
        if invalid:
            raise ValueError(f"matrix row {row_index} is not ternary: {invalid[0]}")
    return rows, width


def validate_integer_matrix(
    matrix: Sequence[Sequence[int]], *, maximum_absolute_weight: int | None = None,
) -> tuple[int, int]:
    """Validate a rectangular signed-integer learned-weight matrix."""

    if not matrix:
        raise ValueError("matrix must have at least one row")
    width = len(matrix[0])
    if width == 0:
        raise ValueError("matrix must have at least one column")
    for row_index, row in enumerate(matrix):
        if len(row) != width:
            raise ValueError(f"matrix row {row_index} has inconsistent width")
        if any(isinstance(weight, bool) or not isinstance(weight, int) for weight in row):
            raise ValueError(f"matrix row {row_index} contains a non-integer weight")
        if maximum_absolute_weight is not None and any(
            abs(weight) > maximum_absolute_weight for weight in row
        ):
            raise ValueError(
                f"matrix row {row_index} exceeds signed weight limit {maximum_absolute_weight}"
            )
    return len(matrix), width


def ternary_linear(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> Vector:
    """Compute a direct-spatial ternary matrix product with int32 wrapping."""

    rows, columns = validate_ternary_matrix(matrix)
    values = _wrapped_vector(vector, width=columns)
    output: Vector = []
    for row in matrix:
        total = 0
        for weight, value in zip(row, values):
            if weight:
                total = add(total, mul(value, weight))
        output.append(total)
    return output


def integer_linear(
    matrix: Sequence[Sequence[int]], vector: Sequence[int],
    *, maximum_absolute_weight: int | None = None,
) -> Vector:
    """Compute a sparse signed-integer matrix product with int32 wrapping."""

    rows, columns = validate_integer_matrix(
        matrix, maximum_absolute_weight=maximum_absolute_weight,
    )
    values = _wrapped_vector(vector, width=columns)
    output: Vector = []
    for row in matrix:
        total = 0
        for weight, value in zip(row, values, strict=True):
            if weight:
                total = add(total, mul(value, weight))
        output.append(total)
    if len(output) != rows:  # pragma: no cover - defensive shape assertion
        raise AssertionError("integer linear output shape changed")
    return output


def rational_scale(vector: Sequence[int], numerator: int = 1, denominator: int = 1) -> Vector:
    """Apply hardware-realizable trunc((x*p)/q) with int32 operations."""

    if denominator == 0:
        raise ValueError("scale denominator cannot be zero")
    return [trunc_div(mul(value, numerator), denominator) for value in vector]


def relu(vector: Sequence[int]) -> Vector:
    return [value if (value := wrap(raw)) > 0 else 0 for raw in vector]


def residual_add(left: Sequence[int], right: Sequence[int]) -> Vector:
    if len(left) != len(right):
        raise ValueError("residual vectors must have the same width")
    return [add(a, b) for a, b in zip(left, right)]


def abs_int32(value: int) -> int:
    """Absolute value realizable by signed combinators, including INT32_MIN."""

    value = wrap(value)
    return value if value >= 0 else neg(value)


def maxnorm(vector: Sequence[int]) -> Vector:
    """Factorio-native MaxNorm: trunc(31*x/max(max(abs(x)), 1))."""

    values = _wrapped_vector(vector)
    if not values:
        raise ValueError("MaxNorm vector cannot be empty")
    maximum = max(abs_int32(value) for value in values)
    denominator = max(maximum, 1)
    return [trunc_div(mul(31, value), denominator) for value in values]


@dataclass
class VectorRegister:
    """Transaction-level model of a signed vector register.

    Hardware tick latency is documented separately.  This emulator models the
    state observed after a COMMIT/RESET transaction has settled.
    """

    width: int = 32
    state: Vector = field(init=False)

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("register width must be positive")
        self.state = [0] * self.width

    def step(
        self,
        data: Sequence[int] | None = None,
        *,
        commit: bool = False,
        reset: bool = False,
    ) -> Vector:
        if reset:
            self.state = [0] * self.width
        elif commit:
            if data is None:
                raise ValueError("COMMIT requires input data")
            self.state = _wrapped_vector(data, width=self.width)
        elif data is not None and len(data) != self.width:
            raise ValueError(f"expected vector width {self.width}, got {len(data)}")
        return self.read()

    def read(self) -> Vector:
        return list(self.state)


@dataclass
class VectorShiftRegister:
    """Age-ordered vector storage: slot 0 newest, slot depth-1 oldest."""

    depth: int
    width: int = 32
    slots: list[Vector] = field(init=False)

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("shift-register depth must be positive")
        if self.width <= 0:
            raise ValueError("shift-register width must be positive")
        self.slots = [[0] * self.width for _ in range(self.depth)]

    def step(
        self,
        data: Sequence[int] | None = None,
        *,
        commit: bool = False,
        reset: bool = False,
    ) -> list[Vector]:
        if reset:
            self.slots = [[0] * self.width for _ in range(self.depth)]
        elif commit:
            if data is None:
                raise ValueError("COMMIT requires input data")
            incoming = _wrapped_vector(data, width=self.width)
            self.slots = [incoming] + [list(slot) for slot in self.slots[:-1]]
        elif data is not None and len(data) != self.width:
            raise ValueError(f"expected vector width {self.width}, got {len(data)}")
        return self.read()

    def read(self) -> list[Vector]:
        return [list(slot) for slot in self.slots]


@dataclass
class KVStorage:
    """One Transformer layer's K/V history foundation."""

    depth: int = 64
    width: int = 32
    keys: VectorShiftRegister = field(init=False)
    values: VectorShiftRegister = field(init=False)
    validity: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        self.keys = VectorShiftRegister(self.depth, self.width)
        self.values = VectorShiftRegister(self.depth, self.width)
        self.validity = [False] * self.depth

    def step(
        self,
        key: Sequence[int] | None = None,
        value: Sequence[int] | None = None,
        *,
        commit: bool = False,
        reset: bool = False,
    ) -> dict[str, list[Vector]]:
        if commit and (key is None or value is None) and not reset:
            raise ValueError("COMMIT requires both K and V vectors")
        self.keys.step(key, commit=commit, reset=reset)
        self.values.step(value, commit=commit, reset=reset)
        if reset:
            self.validity = [False] * self.depth
        elif commit:
            self.validity = [True] + self.validity[:-1]
        return self.read()

    def read(self) -> dict[str, Any]:
        return {
            "keys": self.keys.read(),
            "values": self.values.read(),
            "validity": list(self.validity),
        }
