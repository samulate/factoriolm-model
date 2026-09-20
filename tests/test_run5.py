import unittest
from pathlib import Path

from factorio_lm.checkpoint import ExactCheckpoint


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "generated" / "run5" / "factorio-lm-quality-exact.json"


class Run5ScaledIntegerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT.exists():
            raise unittest.SkipTest("Run 5 exact checkpoint has not been generated")
        cls.checkpoint = ExactCheckpoint.read(CHECKPOINT)

    def test_checkpoint_is_exact_scaled_int3_with_fixed_nonzero_budget(self):
        rendered = self.checkpoint.to_dict()
        self.assertEqual(rendered["version"], 2)
        self.assertEqual(self.checkpoint.weight_limit, 3)
        self.assertEqual(ExactCheckpoint.from_dict(rendered).to_dict(), rendered)
        nonzero = sum(row["negative"] + row["positive"]
                      for row in self.checkpoint.sparsity_report().values())
        self.assertEqual(nonzero, 25_100)


if __name__ == "__main__":
    unittest.main()
