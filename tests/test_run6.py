import importlib.util
from copy import deepcopy
import unittest
from pathlib import Path

from factorio_lm.checkpoint import ExactCheckpoint
from factorio_lm.full_model import ExactFactorioLM
from factorio_lm.run6_pipeline import (
    _compare_general_quality,
    factorio_greedy_tokens_and_margins,
    generate_incremental_integer,
    load_int4_run5_mask,
)


ROOT = Path(__file__).resolve().parents[1]
RUN6 = ROOT / "generated" / "run6"


class QualityComparisonTests(unittest.TestCase):
    def setUp(self):
        self.baseline_metrics = {"accuracy": 0.4, "cross_entropy": 2.0}
        self.candidate_metrics = {"accuracy": 0.5, "cross_entropy": 1.5}
        self.baseline_quality = {
            "Once ": {"dictionary_like_word_fraction": 0.8,
                      "adjacent_repeated_word_fraction": 0.2},
            "Today ": {"dictionary_like_word_fraction": 0.6,
                       "adjacent_repeated_word_fraction": 0.4},
        }
        self.candidate_quality = {
            "Once ": {"dictionary_like_word_fraction": 0.9,
                      "adjacent_repeated_word_fraction": 0.1},
            "Today ": {"dictionary_like_word_fraction": 0.7,
                       "adjacent_repeated_word_fraction": 0.3},
        }

    def compare(self, **overrides):
        inputs = {
            "baseline_metrics": self.baseline_metrics,
            "candidate_metrics": self.candidate_metrics,
            "baseline_quality": self.baseline_quality,
            "candidate_quality": self.candidate_quality,
        }
        return _compare_general_quality(**(inputs | overrides))

    def test_improvement_uses_metrics_and_sample_averages(self):
        report = self.compare()
        self.assertTrue(report["clearly_better_than_run5"])
        evidence = report["evidence"]
        self.assertAlmostEqual(evidence["exact_accuracy_absolute_gain"], 0.1)
        self.assertAlmostEqual(evidence["raw_cross_entropy_reduction"], 0.5)
        self.assertAlmostEqual(evidence["dictionary_word_rate_gain"], 0.1)
        self.assertAlmostEqual(evidence["adjacent_word_repetition_reduction"], 0.1)

    def test_regressions_do_not_pass_despite_other_improvements(self):
        for key, value in (("accuracy", 0.3), ("cross_entropy", 2.1)):
            with self.subTest(metric=key):
                report = self.compare(candidate_metrics=self.candidate_metrics | {key: value})
                self.assertFalse(report["clearly_better_than_run5"])
        for key, value in (("dictionary_like_word_fraction", 0.1),
                           ("adjacent_repeated_word_fraction", 0.9)):
            with self.subTest(quality=key):
                quality = deepcopy(self.candidate_quality)
                quality["Today "][key] = value
                report = self.compare(candidate_quality=quality)
                self.assertFalse(report["clearly_better_than_run5"])

    def test_identical_results_are_not_an_improvement(self):
        report = self.compare(candidate_metrics=self.baseline_metrics,
                              candidate_quality=self.baseline_quality)
        self.assertFalse(report["clearly_better_than_run5"])
        self.assertTrue(all(value == 0 for value in report["evidence"].values()))


class Run6ScaledInt4Tests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_factorio_greedy_tie_break_is_lowest_signal_index(self):
        import torch

        logits = torch.full((2, 64), -10.0)
        logits[0, 7] = logits[0, 3] = 11
        logits[1, 63] = logits[1, 0] = -2
        tokens, margins = factorio_greedy_tokens_and_margins(logits)
        self.assertEqual(tokens.tolist(), [3, 0])
        self.assertEqual(margins.tolist(), [0, 0])


    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_run6_scaled_int4_and_fast_recurrent_path_match_exact_emulator(self):
        checkpoint_path = RUN6 / "factorio-lm-quality-exact.json"
        if not checkpoint_path.exists():
            self.skipTest("Run 6 release artifact has not been generated")
        checkpoint = ExactCheckpoint.read(checkpoint_path)
        self.assertEqual(checkpoint.weight_limit, 7)
        coefficients = [value for layer in checkpoint.layers for name in (
            "query", "key", "value", "output", "ffn_up", "ffn_down")
                        for row in getattr(layer, name) for value in row]
        coefficients.extend(value for row in checkpoint.lm_head for value in row)
        self.assertLessEqual(set(coefficients), set(range(-7, 8)))
        self.assertTrue(any(abs(value) > 3 for value in coefficients))

        _, _, _, model = load_int4_run5_mask()
        prompt = "He loved to play with it every "
        fast = generate_incremental_integer(model, [prompt], 16)[0]
        exact = ExactFactorioLM(checkpoint).generate(prompt, 16)[0]
        self.assertEqual(fast, exact)
        self.assertEqual(exact, "day\nendoftext \nO")


if __name__ == "__main__":
    unittest.main()
