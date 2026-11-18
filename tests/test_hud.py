"""The command center: command routing, demo hold-out and feedback loop, simulator, HTTP routes."""
import io
import json
import unittest
from wsgiref.util import setup_testing_defaults

from pacesense.coach import CoachService
from pacesense.hud import HudSession, route, simulate_race, ui_commands
from pacesense.storage import Repository
from pacesense.web import App
from pacesense.web.steer_api import demo_runner_id

from tests import helpers


def types(actions):
    return [a["type"] for a in actions]


class CommandParsingTests(unittest.TestCase):
    def test_panels_and_perspectives(self):
        a, said, rest = ui_commands("show the simulator and hide the change log")
        self.assertEqual([(x["type"], x["panel"]) for x in a], [("focus", "simulator"), ("close", "log")])
        self.assertEqual(rest, "")
        a, _, _ = ui_commands("switch to runner view")
        self.assertEqual(a, [{"type": "perspective", "value": "runner"}])
        self.assertEqual(ui_commands("researcher mode")[0][0]["value"], "researcher")

    def test_graph_requests_open_the_race_contribution_view(self):
        for phrase in ("show me some graphs", "present the charts", "visualize the prediction breakdown", "plot contributions"):
            actions, said, rest = ui_commands(phrase)
            self.assertEqual(actions, [{"type": "focus", "panel": "contributions"}], phrase)
            self.assertEqual(rest.strip(), "", phrase)
            self.assertIn("opened contributions", said[0])

    def test_analysis_requests_open_insights(self):
        actions, _, rest = ui_commands("show the analysis")
        self.assertEqual(actions, [{"type": "focus", "panel": "analysis"}])
        self.assertEqual(rest.strip(), "")

    def test_simulation_with_conditions(self):
        a, _, rest = ui_commands("simulate 26 degrees with a 15 km/h headwind and she starts too fast")
        self.assertEqual(a, [{"type": "simulate", "temp_c": 26.0, "wind_kmh": 15.0, "profile": "fast_start"}])
        self.assertEqual(rest, "")
        self.assertEqual(ui_commands("what if it's a hot day")[0][0]["temp_c"], 27.0)

    def test_other_commands(self):
        self.assertEqual(types(ui_commands("calm mode please")[0]), ["calm"])
        self.assertEqual(types(ui_commands("reveal")[0]), ["reveal"])
        self.assertEqual(ui_commands("open runner R00042")[0], [{"type": "runner", "id": "R00042"}])
        self.assertEqual(ui_commands("open runner R0004")[0], [{"type": "runner", "id": "R0004"}])
        self.assertEqual(types(ui_commands("reset the layout")[0]), ["layout"])

    def test_runner_database_and_familiarity_commands(self):
        a, said, rest = ui_commands("open the runner database")
        self.assertEqual(a, [{"type": "runner_database"}])
        self.assertEqual(rest, "")
        self.assertIn("existing runner database", said[0])

        a, _, rest = ui_commands("I don't know this runner")
        self.assertEqual(a, [{"type": "condition", "value": "unknown"}])
        self.assertEqual(rest, "")

        a, _, rest = ui_commands("I know this runner")
        self.assertEqual(a, [{"type": "condition", "value": "known"}])
        self.assertEqual(rest, "")

        a, _, rest = ui_commands("this runner is new to me")
        self.assertEqual(a, [{"type": "condition", "value": "unknown"}])
        self.assertEqual(rest, "")

    def test_race_talk_is_left_for_the_interpreter(self):
        a, _, rest = ui_commands("2016 was a hot day and she was injured")
        self.assertEqual(a, [])
        self.assertIn("2016", rest)


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        cls.rid = demo_runner_id(cls.coach)

    def demo(self):
        return HudSession(self.coach, self.rid, holdout=True)

    def test_demo_hides_the_best_race_until_revealed(self):
        s = self.demo()
        st = s.state()
        self.assertEqual(st["mode"], "demo")
        self.assertIsNone(st["holdout"]["truth_s"])
        best = min(r.finish_s for r in self.coach.runner(self.rid).recent(5))
        self.assertNotIn(round(best), [round(r["finish_s"]) for r in st["races"]])
        s.revealed = True
        self.assertAlmostEqual(s.state()["holdout"]["truth_s"], best)

    def test_feedback_card_reports_accuracy_and_next_steps(self):
        s = self.demo()
        slow = next(r for r in s.state()["races"] if r["evidence"]["direction"] > 0)
        r = route(f"{slow['year']} was a hot day and she was injured", s)
        self.assertEqual(r.used, ["steer"])
        card = s.feedback_card(r.turn)
        self.assertEqual(card["items"][0]["verdict"], "supported")
        self.assertIn("accuracy", card)
        self.assertIn(card["accuracy"]["text"], ("closer to her real result", "further from her real result",
                                                 "no closer to her real result"))
        self.assertTrue(card["suggestions"])
        self.assertIn("went on to run", r.reply)                  # the spoken reply gives the reason
        self.assertTrue(card["reasoning"])

    def test_simulating_a_hot_day_never_changes_a_race(self):
        s = self.demo()
        slow = next(r for r in s.state()["races"] if r["evidence"]["direction"] > 0)
        route(f"{slow['year']} was terrible", s)                        # context is now that race
        before = list(s.plan.weights)
        r = route("simulate a hot day", s)
        self.assertEqual(types(r.actions), ["simulate"])
        self.assertEqual(s.plan.weights, before)

    def test_repeating_a_change_says_so(self):
        s = self.demo()
        y = s.state()["races"][0]["year"]
        route(f"ignore {y}", s)
        r = route(f"ignore {y}", s)
        self.assertIn("already counts for nothing", r.reply)

    def test_questions_go_to_the_assistant(self):
        s = self.demo()
        r = route("what is the likely range?", s)
        self.assertEqual(r.used, ["assistant"])
        self.assertTrue(r.reply)

    def test_ai_steering_reads_natural_language_paraphrases(self):
        from types import SimpleNamespace

        class FakeClient:
            def __init__(self, payload):
                self.payload = payload

            def chat(self, *args, **kwargs):
                return SimpleNamespace(text=json.dumps(self.payload))

        class FakeAI:
            online = True

            def __init__(self, payload):
                self.client = FakeClient(payload)

            def config(self, task):
                return "fake-model", 0.0

        s = HudSession(self.coach, self.rid, ai=FakeAI({
            "changes": [{"race": 0, "direction": "down", "strength": "normal",
                         "quote": "should matter as much"}],
            "command": None,
            "estimate": None,
        }))
        race = s.steer.races[0]
        text = f"I don't think {race['event']} {race['year']} is as appealing as the others"
        s.steer.ai.client.payload["changes"][0]["quote"] = "as appealing as the others"
        r = route(text, s)
        self.assertEqual(r.used, ["steer"])
        self.assertEqual(r.turn["applied"][0]["race"], f"{race['event']} {race['year']}")
        self.assertAlmostEqual(r.turn["applied"][0]["new"], 0.3)

    def test_reactions_are_counted(self):
        s = self.demo()
        s.react(1, "agree"); s.react(2, "unsure")
        self.assertEqual(s.reaction_summary(), {"n": 2, "agree": 1, "unsure": 1, "disagree": 0})
        with self.assertRaises(Exception):
            s.react(3, "maybe")


class SimulatorTests(unittest.TestCase):
    PACES = [300.0] * 9

    def test_heat_and_wind_slow_the_runner(self):
        cool = simulate_race(self.PACES, 12000, 13500, temp_c=10)["summary"]["sim_finish_s"]
        hot = simulate_race(self.PACES, 12000, 13500, temp_c=28)["summary"]["sim_finish_s"]
        windy = simulate_race(self.PACES, 12000, 13500, temp_c=10, wind_kmh=30)["summary"]["sim_finish_s"]
        self.assertLess(cool, windy)
        self.assertLess(windy, hot)
        self.assertGreater(hot / cool, 1.03)          # about 0.3 %/°C above 11 °C

    def test_frames_cover_the_race_and_the_band(self):
        sim = simulate_race(self.PACES, 12000, 13500)
        fr = sim["frames"]
        self.assertEqual(fr[0]["km"], 0.0)
        self.assertAlmostEqual(fr[-1]["km"], 42.195)
        self.assertTrue(all(f["low_s"] <= f["plan_s"] <= f["high_s"] for f in fr))
        self.assertIn("assumptions", sim)

    def test_bad_conditions_are_rejected(self):
        with self.assertRaises(Exception):
            simulate_race(self.PACES, None, None, temp_c=60)


def call(app, method, path, body=None):
    env = {}
    setup_testing_defaults(env)
    raw = json.dumps(body).encode() if body is not None else b""
    env.update(REQUEST_METHOD=method, PATH_INFO=path, CONTENT_LENGTH=str(len(raw)))
    env["wsgi.input"] = io.BytesIO(raw)
    st = {}
    out = b"".join(app(env, lambda s, h: st.update(code=int(s.split()[0]), h=dict(h))))
    return st["code"], (json.loads(out) if st["h"]["Content-Type"].startswith("application/json") else out.decode())


class HudWebTests(unittest.TestCase):
    def test_round_trip(self):
        app = App(helpers.service(), ai_factory=lambda: None)
        code, html = call(app, "GET", "/hud")
        self.assertEqual(code, 200)
        self.assertIn("COMMAND CENTER", html)
        code, d = call(app, "POST", "/api/hud/sessions", {"runner_id": "demo"})
        self.assertEqual(code, 201)
        sid = d["id"]
        code, d = call(app, "POST", f"/api/hud/sessions/{sid}/command", {"text": "simulate 25 degrees"})
        self.assertEqual(code, 200)
        self.assertEqual(d["simulation"]["summary"]["temp_c"], 25.0)
        code, d = call(app, "POST", f"/api/hud/sessions/{sid}/weights", {"index": 0, "weight": 0.9})
        self.assertEqual(d["card"]["items"][0]["new"], 0.9)
        code, d = call(app, "POST", f"/api/hud/sessions/{sid}/reveal")
        self.assertIsNotNone(d["holdout"]["truth_s"])
        code, d = call(app, "POST", f"/api/hud/sessions/{sid}/react", {"turn": 1, "value": "agree"})
        self.assertEqual(d["agree"], 1)
        self.assertEqual(call(app, "GET", "/api/hud/track-record")[0], 200)
        self.assertEqual(call(app, "POST", f"/api/hud/sessions/{sid}/command", {"text": ""})[0], 422)
        self.assertEqual(call(app, "GET", "/api/hud/sessions/nosuchsession1")[0], 404)


if __name__ == "__main__":
    unittest.main()
