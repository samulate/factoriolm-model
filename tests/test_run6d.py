from collections import Counter
import importlib.util
from pathlib import Path
import unittest

from factorio_lm.run6d_pipeline import (
    coherent_quality_metrics,
    coherent_quality_score,
    early_quality_metrics,
    hard_early_quality_pass,
    load_clean_int4,
)
from factorio_lm.run6_pipeline import (
    generate_incremental_scored,
    generate_incremental_scored_ragged,
)


ROOT = Path(__file__).resolve().parents[1]


class Run6DEarlyQualityTests(unittest.TestCase):
    def setUp(self):
        self.lexicon = {
            "a", "bird", "found", "little", "red", "ball", "and", "was",
            "very", "happy", "big", "day", "the",
        }

    def test_clean_early_clause_passes(self):
        generated = "a little bird found a red ball and was very happy"
        passed, reasons = hard_early_quality_pass(generated, self.lexicon)
        self.assertTrue(passed, reasons)

    def test_separator_and_word_loop_fail(self):
        generated = "day endoftext big big big big big big big big big big"
        passed, reasons = hard_early_quality_pass(generated, self.lexicon)
        self.assertFalse(passed)
        self.assertIn("endoftext in first 48", reasons)
        self.assertIn("adjacent duplicate word in first 48", reasons)

    def test_repeated_character_run_fails(self):
        generated = "a little bird found a red ballllll and was very happy"
        metrics = early_quality_metrics(generated[:32], self.lexicon)
        self.assertGreater(metrics["maximum_repeated_character_run"], 2)
        self.assertFalse(hard_early_quality_pass(generated, self.lexicon)[0])

    def test_coherence_score_penalizes_repeated_phrase(self):
        corpus = "sam played with his friend and they were very happy"
        words = corpus.split()
        unigrams = Counter(words)
        bigrams = Counter(zip(words, words[1:]))
        trigrams = Counter(zip(words, words[1:], words[2:]))
        ngrams = (unigrams, bigrams, trigrams)
        clean = coherent_quality_metrics(
            "Sam decided to ",
            "play with his friend and they were very happy",
            ngrams,
        )
        loop = coherent_quality_metrics(
            "Sam decided to ",
            "play with his friend play with his friend play with his friend",
            ngrams,
        )
        self.assertGreater(coherent_quality_score(clean), coherent_quality_score(loop))
        self.assertGreater(loop["repeated_word_trigrams"], 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_mixed_length_search_batch_matches_grouped_generation(self):
        artifact = ROOT / "generated" / "run6d" / "factorio-lm-cleaned-int4.pt"
        if not artifact.exists():
            self.skipTest("Run 6D cleaned model has not been generated")
        config, _, _, model = load_clean_int4(artifact.parent)
        prompts = ["Sam decided to ", "Once upon a time a little girl "]
        ragged = generate_incremental_scored_ragged(
            model, prompts, 12, device=config.device)
        grouped = [
            generate_incremental_scored(
                model, [prompt], 12, device=config.device)[0]
            for prompt in prompts
        ]
        self.assertEqual(ragged, grouped)


if __name__ == "__main__":
    unittest.main()
