"""Known inputs for the integer arithmetic and state tests."""

from __future__ import annotations


LINEAR4_MATRIX = [
    [1, 0, -1, 1],
    [0, -1, 1, 0],
    [1, 1, 0, -1],
    [-1, 0, 1, 1],
]
LINEAR4_INPUT = [3, -2, 5, 7]


MAXNORM_INPUT = [
    -93,
    -62,
    -31,
    -15,
    -7,
    -3,
    -1,
    0,
    1,
    2,
    4,
    8,
    16,
    24,
    30,
    31,
    32,
    47,
    62,
    93,
    5,
    -5,
    11,
    -11,
    17,
    -17,
    23,
    -23,
    29,
    -29,
    45,
    -45,
]
REGISTER_INPUT = [(-1 if index % 2 else 1) * (index + 1) for index in range(32)]
K_INPUT = [index - 16 for index in range(32)]
V_INPUT = [16 - index for index in range(32)]

