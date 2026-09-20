import contextlib
import io
import json
import unittest
from unittest.mock import patch

from factorio_lm.cli import main


class CLITests(unittest.TestCase):
    def test_exact_inference_uses_the_included_checkpoint(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["emulate-run3", "--prompt", "Factorio ", "--count", "1"])
        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["generated"], "b")
        self.assertEqual(report["transactions"], len("Factorio "))

    def test_successful_run4_training_finishes_after_reporting_metrics(self):
        metrics = {"cross_entropy": 1.0, "perplexity": 2.0, "accuracy": 0.5}
        report = {
            "metrics": {name: metrics for name in ("float", "qat", "exact")},
            "clearly_better_than_run3": True,
        }
        with patch("factorio_lm.run4_pipeline.run_training",
                   return_value={"report": report}) as train:
            with contextlib.redirect_stdout(io.StringIO()):
                result = main([
                    "train-run4", "--train-source", "train.txt",
                    "--validation-source", "validation.txt",
                ])
        self.assertEqual(result, 0)
        self.assertEqual(train.call_args.kwargs["train_source"], "train.txt")

    def test_run5_evaluation_does_not_require_machine_artifacts(self):
        metrics = {"cross_entropy": 1.0, "perplexity": 2.0, "accuracy": 0.5}
        with patch("factorio_lm.run5_pipeline.finalize_evaluation", return_value={
            "metrics": {"scaled_int3_exact_forward": metrics},
        }) as evaluate:
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(["evaluate-run5", "--output-dir", "/tmp/model-evaluation"])
        self.assertEqual(result, 0)
        self.assertEqual(evaluate.call_args.kwargs["output_dir"], "/tmp/model-evaluation")


if __name__ == "__main__":
    unittest.main()
