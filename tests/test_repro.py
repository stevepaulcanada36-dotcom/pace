"""`pacesense reproduce`: same seed, same outputs; any change shows up in the comparison."""
import json
import tempfile
import unittest
from pathlib import Path

from pacesense.repro import DEFAULT_SEED, Params, compare, configured_seed, run_pipeline


def tiny(seed=5):
    return Params(seed, n_runners=900, validate_n=40, recency_n=30, coaches=6)


class ReproduceTests(unittest.TestCase):
    def test_two_runs_give_identical_outputs(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            m1 = run_pipeline(Path(a), tiny(), log=lambda m: None)
            m2 = run_pipeline(Path(b), tiny(), log=lambda m: None)
            self.assertEqual(compare(m1, m2), [])
            self.assertEqual(set(m1["outputs"]), {"validation.json", "recency_fit.json", "steer_benchmark.json",
                                                  "study_report.md"})
            saved = json.loads((Path(a) / "manifest.json").read_text())
            self.assertEqual(saved["outputs"], m1["outputs"])
            self.assertIn("python", saved["environment"])

    def test_a_different_seed_is_detected(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            m1 = run_pipeline(Path(a), tiny(5), log=lambda m: None)
            m2 = run_pipeline(Path(b), tiny(6), log=lambda m: None)
            diffs = compare(m1, m2)
            self.assertTrue(any(d.startswith("parameters.seed") for d in diffs))
            self.assertTrue(any(d.startswith("inputs.data_sha256") for d in diffs))

    def test_seed_comes_from_config(self):
        self.assertEqual(configured_seed(), DEFAULT_SEED)


if __name__ == "__main__":
    unittest.main()
