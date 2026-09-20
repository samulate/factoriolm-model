import random
import unittest

from factorio_lm.int32 import (
    INT32_MAX,
    INT32_MIN,
    add,
    left_shift,
    mod,
    mul,
    right_shift,
    sub,
    trunc_div,
    wrap,
)


class Int32Tests(unittest.TestCase):
    def test_wrap_boundaries(self):
        self.assertEqual(wrap(0), 0)
        self.assertEqual(wrap(INT32_MAX), INT32_MAX)
        self.assertEqual(wrap(INT32_MAX + 1), INT32_MIN)
        self.assertEqual(wrap(INT32_MIN - 1), INT32_MAX)
        self.assertEqual(wrap((1 << 32) + 7), 7)

    def test_wrapped_arithmetic(self):
        self.assertEqual(add(INT32_MAX, 1), INT32_MIN)
        self.assertEqual(sub(INT32_MIN, 1), INT32_MAX)
        self.assertEqual(mul(1 << 30, 4), 0)
        rng = random.Random(20260815)
        for _ in range(1000):
            left = rng.randrange(-(1 << 40), 1 << 40)
            right = rng.randrange(-(1 << 40), 1 << 40)
            self.assertEqual(add(left, right), wrap(wrap(left) + wrap(right)))
            self.assertEqual(mul(left, right), wrap(wrap(left) * wrap(right)))

    def test_division_truncates_toward_zero(self):
        self.assertEqual(trunc_div(7, 3), 2)
        self.assertEqual(trunc_div(-7, 3), -2)
        self.assertEqual(trunc_div(7, -3), -2)
        self.assertEqual(trunc_div(-7, -3), 2)
        self.assertEqual(trunc_div(-1, 2), 0)

    def test_division_edge_cases(self):
        self.assertEqual(trunc_div(123, 0), 0)
        self.assertEqual(trunc_div(INT32_MIN, -1), INT32_MIN)
        self.assertEqual(mod(-7, 3), -1)
        self.assertEqual(mod(7, -3), 1)
        self.assertEqual(mod(7, 0), 0)

    def test_shifts(self):
        self.assertEqual(left_shift(1, 12), 4096)
        self.assertEqual(left_shift(1 << 30, 1), INT32_MIN)
        self.assertEqual(left_shift(1, 32), 0)
        self.assertEqual(right_shift(-8, 2), -2)
        self.assertEqual(right_shift(-1, 32), -1)
        with self.assertRaises(ValueError):
            left_shift(1, -1)


if __name__ == "__main__":
    unittest.main()

