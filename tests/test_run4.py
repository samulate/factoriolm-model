import unittest
from dataclasses import replace

from factorio_lm.full_model import ExactFactorioLM
from factorio_lm.run4_pipeline import (
    Run4Config,
    create_float_model,
    initialize_qat_from_float,
    normalize_tinystories,
)
from factorio_lm.training import export_model
from factorio_lm.vocabulary import FACTORIOLM_ALPHABET


class Run4DataTests(unittest.TestCase):
    def test_unicode_typography_and_unsupported_characters_normalize_deterministically(self):
        source = "\u201cOnce\u2014upon\u201d a caf\u00e9! Don\u2019t \u2603.\n"
        left, statistics = normalize_tinystories(source)
        right, _ = normalize_tinystories(source)
        self.assertEqual(left, right)
        self.assertEqual(left, "Once upon a cafe\nDont \n")
        self.assertTrue(set(left) <= set(FACTORIOLM_ALPHABET))
        self.assertGreater(statistics["unsupported_removed"], 0)

    def test_document_separator_is_removed_as_a_whole_token(self):
        source = (
            "A story.<|endoftext|>Another story."
            "<endoftext>Third story.<|ENDOFTEXT|>Last story."
        )
        normalized, statistics = normalize_tinystories(source)
        self.assertNotIn("endoftext", normalized.lower())
        self.assertEqual(statistics["document_separators_mapped_to_boundaries"], 3)
        self.assertEqual(
            normalized,
            "A story\nAnother story\nThird story\nLast story\n",
        )


class Run4TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("optional PyTorch training dependency not installed")

    def test_float_projection_is_strict_ternary_and_budgeted(self):
        config = replace(Run4Config(), batch_size=1)
        model = initialize_qat_from_float(create_float_model(config), config)
        checkpoint = export_model(model)
        checkpoint.validate()
        report = checkpoint.sparsity_report()
        aggregate_zero = sum(row["zero"] for row in report.values())
        aggregate_total = sum(row["total"] for row in report.values())
        self.assertAlmostEqual(aggregate_zero / aggregate_total, config.target_sparsity, places=3)

    def test_vectorized_64_character_qat_matches_exact_emulator(self):
        import torch

        config = replace(Run4Config(), batch_size=1)
        model = initialize_qat_from_float(create_float_model(config), config).eval()
        tokens = [index % 64 for index in range(64)]
        checkpoint = export_model(model)
        with torch.no_grad():
            qat = model(torch.tensor([tokens])).round().to(torch.int64).tolist()[0]
        exact = ExactFactorioLM(checkpoint)
        self.assertEqual(qat, [exact.step(token).logits for token in tokens])


if __name__ == "__main__":
    unittest.main()
