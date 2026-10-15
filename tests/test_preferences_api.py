import unittest

from pacesense.coach import DEFAULT_PREFERENCES, CoachService
from pacesense.domain import ValidationError
from pacesense.storage import Repository
from pacesense.web import App

from .helpers import model, runners, service
from .test_study_web import call


def coach():
    return CoachService(model(), runners(), Repository(":memory:"))


class PreferenceTests(unittest.TestCase):
    def test_defaults_then_update(self):
        c = coach()
        self.assertEqual(c.preferences(), DEFAULT_PREFERENCES)
        self.assertIsNone(c.preferences()["experience"])          # first visit: welcome is shown
        out = c.update_preferences({"experience": "experienced", "view": "full", "unit": "mi"})
        self.assertEqual(out["view"], "full")
        self.assertEqual(c.preferences()["unit"], "mi")
        self.assertTrue(c.preferences()["explanations"])           # untouched keys keep their value

    def test_rejects_bad_values(self):
        c = coach()
        for bad in ({"view": "cards"}, {"unit": "m"}, {"colour": "red"}, {}, {"explanations": 1},
                    {"explanations": "yes"}):
            with self.assertRaises(ValidationError, msg=str(bad)):
                c.update_preferences(bad)
        self.assertEqual(c.preferences(), DEFAULT_PREFERENCES)   # nothing was stored


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.c = coach()
        self.r = next(r for r in runners() if len(r.races) >= 4 and all(x.segments_s for x in r.races))

    def test_numbers_identical_text_follows_unit(self):
        km = self.c.predict(self.r.runner_id, unit="km")
        mi = self.c.predict(self.r.runner_id, unit="mi")
        self.assertEqual(km["finish_s"], mi["finish_s"])           # units never change the model
        self.assertEqual(km["paces"], mi["paces"])                 # paces stay in seconds per km
        fast_km = next(s for s in km["scenarios"] if s["key"] == "fast_start")["description"]
        fast_mi = next(s for s in mi["scenarios"] if s["key"] == "fast_start")["description"]
        self.assertEqual(fast_km, "5 s/km faster to 20 km, 5 s/km slower after")
        self.assertEqual(fast_mi, "8 s/mi faster to 20 km, 8 s/mi slower after")   # 5 s/km x 1.609
        self.assertFalse(any("s/km" in t for t in mi["insights"]))

    def test_plan_check_flags_use_unit(self):
        p = self.c.predict(self.r.runner_id)
        chk = self.c.evaluate_plan(self.r.runner_id, None, [x * 0.85 for x in p["paces"]], unit="mi")
        self.assertTrue(chk["flags"])
        self.assertTrue(all("/mi" in f["message"] for f in chk["flags"]))

    def test_bad_unit(self):
        with self.assertRaises(ValidationError):
            self.c.predict(self.r.runner_id, unit="furlong")


class WebTests(unittest.TestCase):
    def setUp(self):
        self.app = App(service())

    def test_preferences_api(self):
        st, prefs = call(self.app, "GET", "/api/coach/preferences")
        self.assertEqual((st, prefs), (200, DEFAULT_PREFERENCES))
        st, out = call(self.app, "PUT", "/api/coach/preferences", {"view": "full"})
        self.assertEqual((st, out["view"]), (200, "full"))
        self.assertEqual(call(self.app, "PUT", "/api/coach/preferences", {"view": "x"})[0], 422)

    def test_prediction_unit_api(self):
        rid = next(r for r in runners() if len(r.races) >= 3).runner_id
        self.assertEqual(call(self.app, "POST", f"/api/coach/runners/{rid}/prediction", {"unit": "mi"})[1]["unit"], "mi")
        self.assertEqual(call(self.app, "POST", f"/api/coach/runners/{rid}/prediction", {"unit": 5})[0], 422)

    def test_api_index_and_no_store(self):
        st, body, headers = call_raw(self.app, "GET", "/")
        self.assertEqual((st, headers["Cache-Control"]), (200, "no-store"))
        self.assertIn(b"/api/coach/runners", body)
        for path in ("/assets/js/app.js", "/assets/dashboard.css", "/coach", "/study"):
            self.assertEqual(call_raw(self.app, "GET", path)[0], 404, path)


def call_raw(app, method, path, extra=None):
    import io
    from wsgiref.util import setup_testing_defaults
    env = {}
    setup_testing_defaults(env)
    env.update(REQUEST_METHOD=method, PATH_INFO=path, QUERY_STRING="", CONTENT_LENGTH="0")
    env["wsgi.input"] = io.BytesIO(b"")
    env.update(extra or {})
    out = {}

    def start(status, headers):
        out["status"], out["headers"] = int(status.split()[0]), dict(headers)
    data = b"".join(app(env, start))
    return out["status"], data, out["headers"]


if __name__ == "__main__":
    unittest.main()
