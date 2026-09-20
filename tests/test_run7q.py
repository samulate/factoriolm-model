import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from factorio_lm.run7q_sequence import SequenceChatCorpus
from factorio_lm.run7q_prompt_miner import is_non_mood_prompt


class Run7QTests(unittest.TestCase):
    def test_sequence_corpus_buckets_equal_prompt_lengths(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            rows = [
                {"question": "Could aliens visit us?", "answer":
                 "They might visit Earth from a distant planet."},
                {"question": "Could rockets reach Mars?", "answer":
                 "A rocket could travel there after a very long flight."},
            ]
            path.write_text("".join(
                json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            corpus = SequenceChatCorpus(
                path, rollout=8, maximum_answer=96, minimum_examples=2)
            self.assertEqual(corpus.examples, 2)
            self.assertTrue(all(
                len({len(prompt) for prompt, _ in bucket}) == 1
                for bucket in corpus.buckets.values()))

    def test_exhaustive_miner_excludes_mood_prompts_only(self):
        self.assertFalse(is_non_mood_prompt("How are you feeling happy today?"))
        self.assertTrue(is_non_mood_prompt(
            "Could an alien build a machine on the moon?"))


if __name__ == "__main__":
    unittest.main()
