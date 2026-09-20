import json
import unittest
from dataclasses import replace
from pathlib import Path

from factorio_lm.block import BlockScales, MatrixScale
from factorio_lm.checkpoint import ExactCheckpoint
from factorio_lm.controller import AutoregressiveController
from factorio_lm.full_model import ExactFactorioLM, first_generation_trace, greedy_argmax
from factorio_lm.vocabulary import FACTORIOLM_ALPHABET, Vocabulary


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "generated/run3/factorio-lm-smoke-exact.json"
FIRST_GENERATION_PATH = ROOT / "generated/run3/first-generation-trace.json"


class Run3VocabularyCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = ExactCheckpoint.read(CHECKPOINT_PATH)

    def test_fixed_vocabulary_and_unknown_mapping(self):
        vocabulary = Vocabulary()
        self.assertEqual(len(FACTORIOLM_ALPHABET), 64)
        self.assertEqual(vocabulary.decode(vocabulary.encode("Factorio 123")), "Factorio 123")
        self.assertEqual(vocabulary.encode("!"), [vocabulary.unknown_id])

    def test_exact_checkpoint_round_trip_and_strict_validation(self):
        rendered = self.checkpoint.to_dict()
        self.assertEqual(ExactCheckpoint.from_dict(rendered).to_dict(), rendered)
        broken = json.loads(json.dumps(rendered))
        broken["layers"][0]["query"][0][0] = 2
        with self.assertRaisesRegex(ValueError, "ternary"):
            ExactCheckpoint.from_dict(broken)
        broken = json.loads(json.dumps(rendered))
        broken["hardware"]["score_divisor"] = 7
        with self.assertRaisesRegex(ValueError, "score_divisor"):
            ExactCheckpoint.from_dict(broken)

    def test_scaled_small_integer_checkpoint_round_trip(self):
        layers = list(self.checkpoint.layers)
        query = [list(row) for row in layers[0].query]
        query[0][0] = 3
        layers[0] = replace(layers[0], query=query)
        scales = [BlockScales() for _ in range(6)]
        scales[0] = replace(scales[0], query=MatrixScale(1, 7))
        scaled = replace(
            self.checkpoint,
            layers=layers,
            weight_limit=3,
            layer_scales=scales,
            lm_head_scale=MatrixScale(2, 5),
        )
        rendered = scaled.to_dict()
        self.assertEqual(rendered["version"], 2)
        self.assertEqual(rendered["hardware"]["weight_limit"], 3)
        self.assertEqual(rendered["layers"][0]["scales"]["query"],
                         {"numerator": 1, "denominator": 7})
        self.assertEqual(ExactCheckpoint.from_dict(rendered).to_dict(), rendered)

    def test_export_contains_only_exact_discrete_values(self):
        values = self.checkpoint.to_dict()
        self.assertEqual(values["format"], "factorio-lm-exact")
        self.assertNotIn("optimizer", values)
        self.assertNotIn("state_dict", values)
        for report in self.checkpoint.sparsity_report().values():
            self.assertGreaterEqual(report["sparsity_percentage"], 40)
            self.assertLessEqual(report["sparsity_percentage"], 60)
        self.assertIn(self.checkpoint.score_divisor, (16, 32, 64, 128, 256, 512))

    def test_qat_forward_equals_independent_exact_emulator(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch training dependency not installed")
        from factorio_lm.training import TrainingConfig, create_model, export_model

        config = TrainingConfig(sequence_length=4, batch_size=1)
        model = create_model(config, 64).eval()
        tokens = [2, 3, 4, 5]
        checkpoint = export_model(model)
        with torch.no_grad():
            qat = model(torch.tensor([tokens])).round().to(torch.int64).tolist()[0]
        exact = ExactFactorioLM(checkpoint)
        self.assertEqual(qat, [exact.step(token).logits for token in tokens])


class Run3ExactModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = ExactCheckpoint.read(CHECKPOINT_PATH)

    def test_six_independent_caches_and_more_than_64_tokens(self):
        model = ExactFactorioLM(self.checkpoint)
        object_ids = {id(layer.cache) for layer in model.layers}
        self.assertEqual(len(object_ids), 6)
        tokens = model.vocabulary.encode(("Factorio circuits " * 5)[:70])
        for token in tokens:
            model.step(token)
        self.assertEqual(model.steps, 70)
        for layer in model.layers:
            self.assertEqual(sum(layer.cache.validity), 64)
            self.assertTrue(all(layer.cache.validity))

    def test_reset_prompt_priming_and_deterministic_feedback(self):
        left = ExactFactorioLM(self.checkpoint)
        right = ExactFactorioLM(self.checkpoint)
        generated_left, traces = left.generate("Factorio ", 10)
        generated_right, _ = right.generate("Factorio ", 10)
        self.assertEqual(generated_left, generated_right)
        self.assertEqual([layer.steps for layer in left.layers], [len("Factorio ") + 9] * 6)
        left.reset()
        self.assertEqual([sum(layer.cache.validity) for layer in left.layers], [0] * 6)
        self.assertEqual(len(traces), len("Factorio ") + 9)

    def test_argmax_tie_contract(self):
        self.assertEqual(greedy_argmax([-9] * 64), 0)
        values = [-20] * 64
        values[9] = values[17] = -3
        self.assertEqual(greedy_argmax(values), 9)

    def test_controller_separates_prompt_from_visible_generation(self):
        controller = AutoregressiveController(ExactFactorioLM(self.checkpoint), "Build ")
        events = controller.run(3)
        prompt_events = events[:len("Build ")]
        self.assertTrue(all(event["source"] == "prompt-rom" for event in prompt_events))
        self.assertTrue(all(not event["input_is_displayed_generated_character"] for event in prompt_events))
        self.assertTrue(all(not event["predicted_token_is_displayed"] for event in prompt_events[:-1]))
        self.assertTrue(prompt_events[-1]["predicted_token_is_displayed"])
        self.assertTrue(all(event["source"] == "autoregressive-feedback" for event in events[len("Build "):]))
        self.assertEqual(len(controller.visible_generated_text), 3)

    def test_first_generation_oracle_is_exact_and_selects_b(self):
        oracle = json.loads(FIRST_GENERATION_PATH.read_text(encoding="utf-8"))
        self.assertEqual(first_generation_trace(self.checkpoint), oracle)
        self.assertEqual(oracle["chosen_token_id"], 3)
        self.assertEqual(oracle["chosen_token"], "b")
        self.assertEqual(oracle["final_layer_6_output"], [
            -113, 9, -232, 56, -198, 271, -331, -157,
            -183, -127, -105, 235, 312, 389, -96, 278,
            -119, -190, -56, 18, -226, -38, 384, 12,
            382, -21, -122, 239, -55, -206, -60, 26,
        ])


if __name__ == "__main__":
    unittest.main()
