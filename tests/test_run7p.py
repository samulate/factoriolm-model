import hashlib
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from factorio_lm.run7p_pipeline import (
    RUN7P_ALPHABET,
    RUN7P_PUNCTUATION,
    Run7PConfig,
    PunctuationAlignedChatCorpus,
    _serialize_split,
    _semantic_demo_score,
    _diverse_topic_names,
    _symbol_report,
    extract_examples_punctuation,
    normalize_tinychat_punctuation,
)
from factorio_lm.vocabulary import FACTORIOLM_ALPHABET, Vocabulary


ROOT = Path(__file__).resolve().parents[1]


class Run7PPunctuationTests(unittest.TestCase):
    def test_vocabulary_replaces_only_the_ten_digit_slots(self):
        self.assertEqual(len(RUN7P_ALPHABET), 64)
        self.assertEqual(len(set(RUN7P_ALPHABET)), 64)
        self.assertEqual(RUN7P_ALPHABET[:54], FACTORIOLM_ALPHABET[:54])
        self.assertEqual(RUN7P_ALPHABET[54:], RUN7P_PUNCTUATION)
        sample = "Hello, don't stop! (Really?) \"Yes.\""
        vocabulary = Vocabulary(RUN7P_ALPHABET)
        self.assertEqual(vocabulary.decode(vocabulary.encode(sample)), sample)

    def test_normalization_preserves_ascii_and_converts_smart_typography(self):
        normalized, statistics = normalize_tinychat_punctuation(
            "“Hello,” she said—don't stop! <|endoftext|> (Okay?)")
        self.assertEqual(normalized, '"Hello," she said-don\'t stop! \n(Okay?)')
        self.assertEqual(statistics["document_separators_mapped_to_boundaries"], 1)
        self.assertNotIn("endoftext", normalized.lower())

    def test_extraction_retains_terminal_punctuation_and_rejects_digits(self):
        examples = extract_examples_punctuation(
            "[INST] How are you today? [/INST] I am well, thank you!", 7,
            Run7PConfig())
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].question, "How are you today?")
        self.assertEqual(examples[0].answer, "I am well, thank you!")
        rejected = extract_examples_punctuation(
            "[INST] Do you have 3 cats? [/INST] Yes, I do.", 8,
            Run7PConfig())
        self.assertEqual(rejected, [])

    def test_serialization_uses_punctuation_ids_and_answer_mask(self):
        examples = extract_examples_punctuation(
            "[INST] Can you help me? [/INST] Yes, I can help you!", 11,
            Run7PConfig())
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "split"
            report = _serialize_split(examples, prefix)
            text = prefix.with_suffix(".txt").read_text(encoding="ascii")
            tokens = prefix.with_suffix(".bin").read_bytes()
            self.assertEqual(text, "Q Can you help me?\nA Yes, I can help you!\n")
            self.assertEqual(Vocabulary(RUN7P_ALPHABET).decode(list(tokens)), text)
            self.assertEqual(report["punctuation_counts"]["?"], 1)
            self.assertEqual(report["punctuation_counts"][","], 1)
            self.assertEqual(report["punctuation_counts"]["!"], 1)
            self.assertEqual(sum(prefix.with_suffix(".mask.bin").read_bytes()),
                             len("Yes, I can help you!"))

    def test_symbol_metrics_separate_punctuation_from_letters(self):
        vocabulary = Vocabulary(RUN7P_ALPHABET)
        targets = vocabulary.encode("a.,?")
        predictions = vocabulary.encode("a..!" )
        report = _symbol_report(targets, predictions)
        self.assertEqual(report["letter_space"]["accuracy"], 1.0)
        self.assertEqual(report["micro"]["targets"], 3)
        self.assertEqual(report["micro"]["correct"], 1)

    def test_semantic_demo_score_rewards_alignment_and_rejects_defects(self):
        known = Counter({
            word: 10 for word in (
                "i am feeling happy today thank you for asking it is a beautiful "
                "day outside also like box"
            ).split()
        })
        question = "How are you feeling today on this wonderful day?"
        reference = (
            "I am feeling great today, thank you for asking and for chatting with me.")
        coherent, coherent_metrics = _semantic_demo_score(
            question, reference,
            "I am feeling happy today, thank you for asking, it is a beautiful day outside.",
            known)
        irrelevant, irrelevant_metrics = _semantic_demo_score(
            question, reference, "I also like a box today, thank you for asking.", known)
        repeated, repeated_metrics = _semantic_demo_score(
            question, reference,
            "I am feeling happy happy today, thank you for asking.", known)
        self.assertGreater(coherent, irrelevant)
        self.assertTrue(coherent_metrics["expanded_hard_filter_pass"])
        self.assertGreater(
            coherent_metrics["reference_content_word_f1"],
            irrelevant_metrics["reference_content_word_f1"])
        self.assertFalse(repeated_metrics["expanded_hard_filter_pass"])

    def test_diverse_theme_detection_covers_unusual_prompts(self):
        self.assertEqual(
            _diverse_topic_names("Can a glowing fish survive in dark space?"),
            ["animals", "space_mystery", "fear_horror"])
        self.assertEqual(
            _diverse_topic_names("Can a machine make music during a storm?"),
            ["weather_nature", "machines_adventure", "books_creativity"])

    def test_aligned_corpus_minimum_is_parameterized_for_small_validation_views(self):
        with tempfile.TemporaryDirectory() as directory:
            records = Path(directory) / "records.jsonl"
            records.write_text("".join(
                json.dumps({
                    "question": f"Can you help me friend {letter}?",
                    "answer": "Yes, I can help you.",
                }) + "\n"
                for letter in "abcdefghij"
            ), encoding="utf-8")
            corpus = PunctuationAlignedChatCorpus(
                records, maximum_answer=64, minimum_examples=10)
            self.assertEqual(corpus.examples, 10)


    def test_run7_checkpoint_and_selected_demo_are_unchanged(self):
        expected = {
            "factorio-lm-tinychat-exact.json":
                "b96db3108d74b1604565f31697a8465a537cb31c4f55634b4825c8ec233b67f5",
            "selected-demo.json":
                "feeae4b724997f95e0a0e235a4b88417d7d1f881ba75e032ef59c7406b4ad048",
        }
        run7 = ROOT / "generated" / "run7"
        if not all((run7 / name).exists() for name in expected):
            self.skipTest("Run 7 release artifacts are not available")
        actual = {
            name: hashlib.sha256((run7 / name).read_bytes()).hexdigest()
            for name in expected
        }
        self.assertEqual(actual, expected)

if __name__ == "__main__":
    unittest.main()
