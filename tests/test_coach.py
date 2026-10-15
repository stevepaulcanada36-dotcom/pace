import unittest

import numpy as np

from pacesense.cbr import validate_model
from pacesense.coach import CoachService, finish_from_paces
from pacesense.domain import Race, ValidationError
from pacesense.storage import Repository
from pacesense.study import NotFound

from .helpers import model, runners
from .test_study_web import call
from pacesense.web import App
from .helpers import service


def coach():
    return CoachService(model(), runners(), Repository(":memory:"))


class CoachServiceTests(unittest.TestCase):
    def setUp(self):
        self.c = coach()
        self.r = next(r for r in runners() if len(r.races) >= 4 and all(x.segments_s for x in r.races))

    def test_prediction_has_range_and_consistent_plan(self):
        p = self.c.predict(self.r.runner_id)
        self.assertLess(p["low_s"], p["finish_s"])
        self.assertLess(p["finish_s"], p["high_s"])
        self.assertAlmostEqual(finish_from_paces(p["paces"]), p["finish_s"], delta=1)   # plan adds up to the prediction
        self.assertEqual(p["finish_s"], p["default_finish_s"])
        self.assertEqual([s["key"] for s in p["scenarios"]], ["model", "even", "fast_start", "conservative", "negative"])
        even = p["scenarios"][1]
        self.assertAlmostEqual(even["finish_s"], p["finish_s"], delta=1)
        self.assertTrue(p["insights"])

    def test_weights_change_prediction(self):
        n = len(self.r.recent(5))
        w = [0.0] * n; w[0] = 1.0
        p = self.c.predict(self.r.runner_id, w)
        self.assertNotAlmostEqual(p["finish_s"], p["default_finish_s"], delta=0.5)

    def test_plan_check(self):
        p = self.c.predict(self.r.runner_id)
        same = self.c.evaluate_plan(self.r.runner_id, None, p["paces"])
        self.assertEqual(same["verdict"], "realistic")
        self.assertEqual(same["flags"], [])               # the model's own plan never warns
        one = list(p["paces"]); one[0] *= 0.85            # push only the first interval hard
        pushed = self.c.evaluate_plan(self.r.runner_id, None, one)
        self.assertEqual([f["interval"] for f in pushed["flags"]], ["0-5"])
        self.assertAlmostEqual(same["cumulative_s"][-1], same["finish_s"], delta=0.01)
        fast = self.c.evaluate_plan(self.r.runner_id, None, [x - 30 for x in p["paces"]])
        self.assertEqual(fast["verdict"], "ambitious")
        slow = self.c.evaluate_plan(self.r.runner_id, None, [x + 30 for x in p["paces"]])
        self.assertEqual(slow["verdict"], "cautious")
        with self.assertRaises(ValidationError):
            self.c.evaluate_plan(self.r.runner_id, None, [300] * 8)
        with self.assertRaises(ValidationError):
            self.c.evaluate_plan(self.r.runner_id, None, [60] * 9)

    def test_own_runner_plans_and_track_record(self):
        added = self.c.add_runner("Anna", "F", 32, [Race("Amsterdam", 2023, 12948), Race("Rotterdam", 2024, 12791)])
        rid = added["runner_id"]
        self.assertTrue(any(x["runner_id"] == rid for x in self.c.search("")))
        ov = self.c.overview(rid)
        self.assertEqual(len(ov["history"]), 1)
        p = self.c.predict(rid)
        plan = self.c.save_plan(rid, "Rotterdam 2025", "2025-04-13", None, p["paces"], "note")
        self.assertEqual(self.c.track_record(), {"n": 0})
        done = self.c.record_result(plan["id"], 12600)
        self.assertIn("errors_pct", done)
        tr = self.c.track_record()
        self.assertEqual(tr["n"], 1)
        self.assertAlmostEqual(tr["plan_error_pct"], 100 * abs(12600 - plan["finish_s"]) / 12600)
        with self.assertRaises(ValidationError):
            self.c.record_result(plan["id"], 100)
        with self.assertRaises(NotFound):
            self.c.runner("nope")
        with self.assertRaises(ValidationError):
            self.c.save_plan(rid, " ", None, None, p["paces"])

    def test_range_is_calibrated(self):
        v = validate_model(model(), runners(), n=400, seed=5)
        self.assertGreater(v.range_coverage_pct, 70)
        self.assertLess(v.range_coverage_pct, 90)


class CoachApiTests(unittest.TestCase):
    def test_flow(self):
        app = App(service())
        status, index = call(app, "GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("/api/coach/preferences", index["endpoints"])
        self.assertEqual(call(app, "GET", "/coach")[0], 404)          # the HTML dashboard moved to Streamlit
        st, r = call(app, "POST", "/api/coach/runners", {"name": "Ann", "gender": "F", "age": 30,
                                                         "races": [{"event": "A", "year": 2022, "finish": "3:40:00"},
                                                                   {"event": "B", "year": 2023, "finish": "3:35:00"}]})
        self.assertEqual(st, 201, r)
        rid = r["runner_id"]
        st, p = call(app, "POST", f"/api/coach/runners/{rid}/prediction", {"weights": [0.2, 1.0]})
        self.assertEqual(st, 200, p)
        st, chk = call(app, "POST", f"/api/coach/runners/{rid}/plan-check", {"paces": p["paces"], "weights": [0.2, 1.0]})
        self.assertEqual(st, 200)
        st, plan = call(app, "POST", "/api/coach/plans", {"runner_id": rid, "target_race": "X", "paces": p["paces"]})
        self.assertEqual(st, 201, plan)
        st, res = call(app, "PUT", f"/api/coach/plans/{plan['id']}/result", {"finish": "3:31:00"})
        self.assertEqual(st, 200, res)
        self.assertEqual(call(app, "GET", "/api/coach/track-record")[1]["n"], 1)
        self.assertEqual(call(app, "POST", f"/api/coach/runners/{rid}/prediction", {"weights": "x"})[0], 422)
        self.assertEqual(call(app, "POST", "/api/coach/runners", {"name": "", "gender": "F", "races": []})[0], 422)
        self.assertEqual(call(app, "GET", "/api/coach/runners/does-not-exist")[0], 404)


if __name__ == "__main__":
    unittest.main()
