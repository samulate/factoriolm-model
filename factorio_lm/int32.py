"""Exact signed 32-bit operations used by Factorio circuit networks.

All public arithmetic operations wrap their result into the signed int32
domain.  Division truncates toward zero; division by zero follows Factorio's
arithmetic-combinator behavior and produces zero.
"""

from __future__ import annotations

INT32_BITS = 32
INT32_MODULUS = 1 << INT32_BITS
INT32_MASK = INT32_MODULUS - 1
INT32_SIGN = 1 << (INT32_BITS - 1)
INT32_MIN = -INT32_SIGN
INT32_MAX = INT32_SIGN - 1


def wrap(value: int) -> int:
    """Wrap *value* to Factorio's signed 32-bit signal range."""

    unsigned = int(value) & INT32_MASK
    return unsigned - INT32_MODULUS if unsigned & INT32_SIGN else unsigned


def add(left: int, right: int) -> int:
    return wrap(wrap(left) + wrap(right))


def sub(left: int, right: int) -> int:
    return wrap(wrap(left) - wrap(right))


def mul(left: int, right: int) -> int:
    return wrap(wrap(left) * wrap(right))


def neg(value: int) -> int:
    return wrap(-wrap(value))


def trunc_div(dividend: int, divisor: int) -> int:
    """Signed integer division with truncation toward zero.

    Factorio arithmetic combinators return zero for division by zero.  The
    only mathematically overflowing int32 quotient, INT32_MIN / -1, wraps
    back to INT32_MIN like every other circuit result.
    """

    dividend = wrap(dividend)
    divisor = wrap(divisor)
    if divisor == 0:
        return 0
    magnitude = abs(dividend) // abs(divisor)
    quotient = -magnitude if (dividend < 0) != (divisor < 0) else magnitude
    return wrap(quotient)


def mod(dividend: int, divisor: int) -> int:
    """Factorio-compatible remainder, paired with truncating division."""

    dividend = wrap(dividend)
    divisor = wrap(divisor)
    if divisor == 0:
        return 0
    return wrap(dividend - trunc_div(dividend, divisor) * divisor)


def bit_and(left: int, right: int) -> int:
    return wrap((left & INT32_MASK) & (right & INT32_MASK))


def bit_or(left: int, right: int) -> int:
    return wrap((left & INT32_MASK) | (right & INT32_MASK))


def bit_xor(left: int, right: int) -> int:
    return wrap((left & INT32_MASK) ^ (right & INT32_MASK))


def left_shift(value: int, amount: int) -> int:
    """Wrap a left shift; Run 2 attention only uses amounts 0 through 12."""

    if amount < 0:
        raise ValueError("shift amount must be non-negative")
    if amount >= INT32_BITS:
        return 0
    return wrap((value & INT32_MASK) << amount)


def right_shift(value: int, amount: int) -> int:
    """Arithmetic signed right shift for non-negative shift amounts."""

    if amount < 0:
        raise ValueError("shift amount must be non-negative")
    if amount >= INT32_BITS:
        return -1 if wrap(value) < 0 else 0
    return wrap(value) >> amount


def clamp(value: int, minimum: int, maximum: int) -> int:
    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    return min(max(wrap(value), minimum), maximum)


def sum_int32(values: list[int] | tuple[int, ...]) -> int:
    total = 0
    for value in values:
        total = add(total, value)
    return total

