from pathlib import Path
import unittest

from factorio_lm.run7_pipeline import (
    ChatExample,
    Run7Config,
    _serialize_split,
    deterministic_page_offsets,
    extract_examples,
)
from factorio_lm.vocabulary import FACTORIOLM_ALPHABET


class Run7DataTests(unittest.TestCase):
    def test_extracts_complete_question_and_answer_before_normalization(self):
        source = (
            "[INST] I had a hard day. What should I do now? [/INST] "
            "You can rest and talk to a friend. That may help. "
            "[INST] This turn has no question. [/INST] It is ignored."
        )
        examples = extract_examples(source, 17, Run7Config())
        self.assertEqual(examples, [ChatExample(
            source_row=17,
            turn=0,
            question="What should I do now",
            answer="You can rest and talk to a friend",
        )])
        self.assertEqual(examples[0].prompt, "Q What should I do now\nA ")
        self.assertLessEqual(len(examples[0].prompt), 64)
        self.assertLessEqual(set(examples[0].record), set(FACTORIOLM_ALPHABET))

    def test_tinychat_markers_cannot_leak_into_serialized_text(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "[INST] Can you help me? [/INST] Yes, I can help you!"
            example = extract_examples(source, 1, Run7Config())[0]
            report = _serialize_split([example], root / "tinychat-train")
            text = (root / "tinychat-train.txt").read_text(encoding="ascii")
            self.assertNotIn("INST", text)
            self.assertEqual(text, "Q Can you help me\nA Yes I can help you\n")
            self.assertEqual(report["answer_characters"], len("Yes I can help you"))
            mask = (root / "tinychat-train.mask.bin").read_bytes()
            self.assertEqual(mask[:len(example.prompt)], b"\0" * len(example.prompt))
            self.assertEqual(mask[len(example.prompt):-1], b"\1" * len(example.answer))

    def test_page_offsets_are_deterministic_unique_and_cover_split(self):
        offsets = deterministic_page_offsets(1_000_000, 200_000, 100)
        self.assertEqual(len(offsets), 2_000)
        self.assertEqual(len(offsets), len(set(offsets)))
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[-1], 999_900)
        self.assertTrue(all(offset % 100 == 0 for offset in offsets))

if __name__ == "__main__":
    unittest.main()
