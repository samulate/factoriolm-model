import unittest

from factorio_lm.block import TransformerBlockEmulator, deterministic_token_inputs


class TransformerBlockEmulatorTests(unittest.TestCase):
    def test_causal_current_token_and_age_order_across_steps(self):
        model = TransformerBlockEmulator()
        traces = [model.step(vector) for vector in deterministic_token_inputs(4)]
        for step, trace in enumerate(traces):
            self.assertEqual(sum(trace.cache_validity), step + 1)
            self.assertEqual(trace.cache_keys[0], trace.key)
            self.assertEqual(trace.cache_values[0], trace.value)
            self.assertTrue(trace.attention.cells[0].valid)
            self.assertEqual(trace.attention.cells[0].dots,
                             traces[step].attention.cells[0].dots)
        self.assertEqual(traces[3].cache_keys[1], traces[2].key)
        self.assertEqual(traces[3].cache_keys[2], traces[1].key)
        self.assertEqual(traces[3].cache_keys[3], traces[0].key)

    def test_reset_clears_validity_and_restarts_age_zero(self):
        model = TransformerBlockEmulator()
        first, second = deterministic_token_inputs(2)
        model.step(first)
        model.reset()
        trace = model.step(second)
        self.assertEqual(trace.step, 0)
        self.assertEqual(trace.cache_validity, [True] + [False] * 63)
        self.assertEqual(trace.cache_keys[0], trace.key)

    def test_deterministic_exact_multi_token_trace(self):
        inputs = deterministic_token_inputs(3)
        first = TransformerBlockEmulator()
        second = TransformerBlockEmulator()
        traces_a = [first.step(vector).to_dict() for vector in inputs]
        traces_b = [second.step(vector).to_dict() for vector in inputs]
        self.assertEqual(traces_a, traces_b)
        self.assertEqual(len(traces_a[-1]["block_output"]), 32)
        self.assertEqual(len(traces_a[-1]["ffn_up"]), 64)

    def test_fixture_sparsity_is_in_target_band(self):
        reports = TransformerBlockEmulator().weights.sparsity_report()
        for report in reports.values():
            self.assertGreaterEqual(report["zero_percentage"], 40)
            self.assertLessEqual(report["zero_percentage"], 60)


if __name__ == "__main__":
    unittest.main()

