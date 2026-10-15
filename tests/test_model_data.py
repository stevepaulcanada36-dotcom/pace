import tempfile
import unittest
from pathlib import Path

import numpy as np

from pacesense.cbr import MarathonModel, holdout_best, validate_model
from pacesense.data import load_csv, runners_to_frame
from pacesense.domain import ValidationError

from .helpers import model, runners


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model, self.runners = model(), runners()
        self.r = next(r for r in self.runners if len(r.races) >= 4)

    def test_prediction_is_plausible(self):
        pred = self.model.predict(self.r)
        self.assertTrue(7200 < pred.finish_s < 25200)
        self.assertEqual(len(pred.profile), 9)
        self.assertEqual(pred.weights, (0.5,) * len(self.r.recent(5)))

    def test_weights_select_races(self):
        races = self.r.recent(5)
        per = self.model.per_race(self.r, races)
        w = [0.0] * len(races); w[0] = 1.0
        self.assertAlmostEqual(MarathonModel.combine(per, w).finish_s, per[0].finish_s)
        with self.assertRaises(ValidationError):
            MarathonModel.combine(per, [0.0] * len(races))
        with self.assertRaises(ValidationError):
            MarathonModel.combine(per, [1.5] * len(races))
        with self.assertRaises(ValidationError):
            MarathonModel.combine(per, [0.5])

    def test_own_cases_are_excluded(self):
        per = self.model.case_base.predict_race(self.r.races[0], self.r.gender, exclude_runner=self.r.runner_id)
        self.assertIsNotNone(per)

    def test_holdout(self):
        best, shown = holdout_best(self.r)
        self.assertNotIn(best, shown)
        self.assertTrue(all(best.finish_s <= x.finish_s for x in shown))

    def test_validation_accuracy(self):
        v = validate_model(self.model, self.runners, n=200)
        self.assertGreater(v.n, 150)
        self.assertLess(v.mean_error_pct, 5.0)


class DataTests(unittest.TestCase):
    def test_csv_round_trip(self):
        rs = runners()[:200]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "races.csv"
            runners_to_frame(rs).to_csv(p, index=False)
            back, rep = load_csv(p)
        self.assertEqual(rep.runners, 200)
        self.assertEqual(rep.rows_skipped, 0)
        a = {r.runner_id: r for r in rs}
        for r in back[:20]:
            self.assertEqual(len(r.races), len(a[r.runner_id].races))
            self.assertTrue(np.allclose(r.races[0].profile(), a[r.runner_id].races[0].profile(), atol=1e-3))

    def test_csv_bad_rows_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text("runner_id,gender,event,year,finish\nA,F,Ams,2018,3:30:00\nA,F,Ams,2019,0:30:00\n,M,R,2019,3:00:00\n")
            rs, rep = load_csv(p)
        self.assertEqual((rep.runners, rep.races, rep.rows_skipped), (1, 1, 2))

    def test_csv_requires_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text("who,when\n1,2\n")
            with self.assertRaises(ValidationError):
                load_csv(p)


if __name__ == "__main__":
    unittest.main()
