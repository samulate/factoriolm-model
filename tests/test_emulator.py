import unittest

from factorio_lm.emulator import (
    abs_int32,
    maxnorm,
    relu,
    residual_add,
    ternary_linear,
    integer_linear,
    rational_scale,
    validate_ternary_matrix,
)
from factorio_lm.fixtures import (
    LINEAR4_INPUT,
    LINEAR4_MATRIX,
    MAXNORM_INPUT,
)
from factorio_lm.int32 import INT32_MAX, INT32_MIN


class PrimitiveEmulatorTests(unittest.TestCase):
    def test_small_signed_integer_linear_and_rational_scale(self):
        matrix = [[1, 2, -3, 7], [-7, 3, -2, -1]]
        self.assertEqual(integer_linear(matrix, [5, -4, 3, 2], maximum_absolute_weight=7),
                         [2, -55])
        self.assertEqual(rational_scale([7, -7, 8, -8], 3, 4), [5, -5, 6, -6])

    def test_ternary_linear_known_answer(self):
        self.assertEqual(ternary_linear(LINEAR4_MATRIX, LINEAR4_INPUT), [5, 7, -6, 9])

    def test_linear_wraps_network_accumulation(self):
        matrix = [[1, 1], [-1, 1]]
        self.assertEqual(
            ternary_linear(matrix, [INT32_MAX, 1]),
            [INT32_MIN, -2147483646],
        )

    def test_matrix_validation(self):
        with self.assertRaises(ValueError):
            validate_ternary_matrix([])
        with self.assertRaises(ValueError):
            validate_ternary_matrix([[1, 0], [1]])
        with self.assertRaises(ValueError):
            validate_ternary_matrix([[2]])

    def test_relu_and_residual(self):
        self.assertEqual(relu([-3, 0, 7]), [0, 0, 7])
        self.assertEqual(residual_add([INT32_MAX, -2], [1, 5]), [INT32_MIN, 3])
        with self.assertRaises(ValueError):
            residual_add([1], [1, 2])

    def test_maxnorm_known_answer(self):
        expected = [
            -31, -20, -10, -5, -2, -1, 0, 0, 0, 0, 1, 2, 5, 8, 10, 10,
            10, 15, 20, 31, 1, -1, 3, -3, 5, -5, 7, -7, 9, -9, 15, -15,
        ]
        self.assertEqual(maxnorm(MAXNORM_INPUT), expected)
        self.assertEqual(maxnorm([0, 0]), [0, 0])

    def test_int32_min_abs_matches_realizable_hardware(self):
        self.assertEqual(abs_int32(INT32_MIN), INT32_MIN)
        self.assertEqual(maxnorm([INT32_MIN]), [INT32_MIN])


if __name__ == "__main__":
    unittest.main()
