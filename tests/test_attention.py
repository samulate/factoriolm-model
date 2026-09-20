import unittest

from factorio_lm.attention import (
    attention_bank,
    attention_cell,
    deterministic_relative_biases,
)


class AttentionEmulatorTests(unittest.TestCase):
    def test_four_head_dots_and_negative_truncation(self):
        query = [1, -2, 3, -4, 5, -6, 7, -8] * 4
        key = [2] * 8 + [-2] * 8 + [1] * 8 + [-1] * 8
        value = list(range(-16, 16))
        trace = attention_cell(
            query, key, value, valid=True, relative_bias=[0, 0, 0, 0], score_divisor=7
        )
        self.assertEqual(trace.dots, [-8, 8, -4, 4])
        self.assertEqual(trace.scaled_scores, [-1, 1, 0, 0])
        self.assertEqual(trace.weights, [32, 128, 64, 64])

    def test_all_score_shift_cases_and_clamp(self):
        zero = [0] * 32
        for requested in range(-9, 10):
            trace = attention_cell(
                zero, zero, zero, valid=True,
                relative_bias=[requested] * 4, score_divisor=1,
            )
            score = min(max(requested, -6), 6)
            self.assertEqual(trace.scores, [score] * 4)
            self.assertEqual(trace.weights, [1 << (score + 6)] * 4)

    def test_invalid_slot_masks_both_reductions(self):
        trace = attention_cell(
            [9] * 32, [7] * 32, [-11] * 32, valid=False,
            relative_bias=[6] * 4, score_divisor=3,
        )
        self.assertTrue(all(weight > 0 for weight in trace.raw_weights))
        self.assertEqual(trace.denominator_contributions, [0] * 4)
        self.assertEqual(trace.weighted_values, [0] * 32)

    def test_64_way_reduction_ignores_invalid_slots(self):
        query = [1] * 32
        keys = [[age % 3 - 1] * 32 for age in range(64)]
        values = [[age - channel for channel in range(32)] for age in range(64)]
        validity = [age < 3 for age in range(64)]
        biases = deterministic_relative_biases()
        full = attention_bank(query, keys, values, validity,
                              relative_biases=biases, score_divisor=8)
        short = attention_bank(query, keys[:3], values[:3], [True] * 3,
                               relative_biases=[row[:3] for row in biases], score_divisor=8)
        self.assertEqual(full.denominators, short.denominators)
        self.assertEqual(full.numerators, short.numerators)
        self.assertEqual(full.output, short.output)

    def test_shape_and_divisor_validation(self):
        with self.assertRaises(ValueError):
            attention_cell([0] * 31, [0] * 32, [0] * 32, valid=True,
                           relative_bias=[0] * 4, score_divisor=1)
        with self.assertRaises(ValueError):
            attention_cell([0] * 32, [0] * 32, [0] * 32, valid=True,
                           relative_bias=[0] * 4, score_divisor=0)


if __name__ == "__main__":
    unittest.main()

