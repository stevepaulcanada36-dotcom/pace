"""The voice assistant: known / unknown runners, loops that switch on and off, answering questions by voice,
persona from how the coach speaks, turn-taking, and the spoken replies."""
import unittest

from pacesense.adaptive import AdaptiveEngine, Event, Persona
from pacesense.adaptive.answers import is_bare_answer, match_choice
from pacesense.adaptive.explain import explain
from pacesense.adaptive.loops import ClarifyLoop, LoopContext, SessionContext
from pacesense.coach import CoachService
from pacesense.storage import Repository
from pacesense.studio import Settings, StudioService
from pacesense.studio.app import create_app
from tests import helpers
from tests.asgi import Client
from tests.test_adaptive import STATE, applied, utter

RACES = [{"label": "Amsterdam 2009"}, {"label": "Amsterdam 2010"}, {"label": "Rotterdam 2014"},
         {"label": "None of them", "action": "dismiss"}]
YES_NO = [{"label": "Yes, count it less", "yes": True}, {"label": "No, keep it as it is", "no": True, "action": "dismiss"},
          {"label": "Why?", "action": "explain"}]


class SpokenAnswerTests(unittest.TestCase):
    def test_yes_no_ordinals_and_names(self):
        cases = [("yes", YES_NO, 0), ("yeah, go ahead", YES_NO, 0), ("sure", YES_NO, 0), ("no", YES_NO, 1),
                 ("not sure", YES_NO, 1), ("why", YES_NO, 2), ("2010", RACES, 1), ("the one in 2014", RACES, 2),
                 ("Rotterdam", RACES, 2), ("the second one", RACES, 1), ("number 1", RACES, 0),
                 ("the last one", RACES, 3), ("none of them", RACES, 3)]
        for text, choices, want in cases:
            with self.subTest(text=text):
                self.assertIs(match_choice(text, choices), choices[want])

    def test_not_an_answer(self):
        self.assertIsNone(match_choice("Amsterdam", RACES))                    # two Amsterdams: ambiguous
        self.assertIsNone(match_choice("yes", RACES))                          # nothing to say yes to
        self.assertIsNone(match_choice("she was injured in Berlin and it was a very hot day overall", RACES))
        self.assertIsNone(match_choice("", RACES))
        self.assertTrue(is_bare_answer("yes please"))
        self.assertFalse(is_bare_answer("yes she was injured in 2010"))


class ConditionTests(unittest.TestCase):
    def engine(self, condition):
        e = AdaptiveEngine(Repository(":memory:"), f"c-{condition}")
        r = e.start({**STATE, "session": {"source_runner": "R1"}}, condition)
        return e, r

    @staticmethod
    def active(r):
        return {lp["key"] for lp in r["loops"] if lp["active"]}

    def test_known_and_unknown_switch_different_loops_on(self):
        _, known = self.engine("known")
        _, unknown = self.engine("unknown")
        _, ask = self.engine("ask")
        self.assertIn("knowledge", self.active(known))
        self.assertIn("bias", self.active(known))
        self.assertNotIn("suggest", self.active(known))
        self.assertNotIn("clarify", self.active(known))
        self.assertIn("suggest", self.active(unknown))
        self.assertIn("clarify", self.active(unknown))
        self.assertNotIn("knowledge", self.active(unknown))
        self.assertNotIn("bias", self.active(unknown))                       # until the coach gives an estimate
        self.assertIn("interview", self.active(ask))
        self.assertTrue(all(lp["reason"] for lp in known["loops"]))
        self.assertEqual(ask["prompt"]["data"]["question"]["id"], "familiarity")

    def test_unknown_runner_the_data_leads(self):
        e, r = self.engine("unknown")
        self.assertEqual(r["prompt"]["loop"], "suggest")
        self.assertIn("Amsterdam 2010", r["prompt"]["message"])               # the strongest outlier first
        yes = r["prompt"]["data"]["choices"][0]
        self.assertTrue(yes["yes"])
        self.assertEqual(yes["say"], "set Amsterdam 2010 to 0.2")
        r = e.process(utter("set Amsterdam 2010 to 0.2", applied=[applied()], prompt_answer="suggest"))
        self.assertIn("Rotterdam 2014", r["prompt"]["message"])               # then the next one
        r = e.process(Event("prompt_answer", data={"loop": "suggest", "choice": "dismiss"}))
        self.assertTrue(any("everything the data flags" in a["message"] for a in r["adaptations"]))

    def test_known_runner_knowledge_is_kept_across_sessions(self):
        repo = Repository(":memory:")
        e = AdaptiveEngine(repo, "c-k")
        e.start({**STATE, "session": {"source_runner": "R1"}}, "known")
        r = e.process(utter("she had a knee injury that spring"))
        self.assertEqual(r["prompt"]["loop"], "knowledge")
        self.assertEqual(r["prompt"]["data"]["choices"][1]["say"], "Amsterdam 2010 she was injured")
        r = e.process(utter("she has been training really well this block"))
        self.assertEqual(r["prompt"]["data"]["choices"][0]["say"], "trust the recent races more")
        e2 = AdaptiveEngine(repo, "c-k")
        r = e2.start({**STATE, "session": {"source_runner": "R1"}}, "known")
        msg = next(a["message"] for a in r["adaptations"] if a["loop"] == "knowledge")
        self.assertIn("knee injury", msg)
        self.assertIn("training really well", msg)

    def test_trial_run_injury_is_not_misassigned_to_a_marathon(self):
        ctx = LoopContext(Persona(), {}, {}, SessionContext(
            state={"races": [{"event": "Amsterdam", "year": 2016}]}, condition="unknown"))
        out = ClarifyLoop().observe(utter("She did a trial run in 2026 and had a leg injury"), ctx)
        self.assertEqual(out, [])

    def test_switching_condition_mid_session(self):
        e, r = self.engine("known")
        r = e.set_condition("unknown")
        self.assertEqual(r["ui"]["condition"], "unknown")
        self.assertIn("suggest", self.active(r))
        self.assertEqual(r["prompt"]["loop"], "suggest")
        with self.assertRaises(ValueError):
            e.set_condition("maybe")

    def test_interview_answer_sets_the_condition(self):
        e, _ = self.engine("ask")
        r = e.process(Event("interview_answer", data={"question": "familiarity", "choice": "none", "familiarity": 0.1}))
        self.assertEqual(r["ui"]["condition"], "unknown")
        self.assertEqual(r["prompt"]["loop"], "suggest")


class VoicePersonaTests(unittest.TestCase):
    def test_how_they_speak_shapes_the_persona(self):
        p = Persona()
        for _ in range(3):
            p.observe(utter("2010 was hot", "voice", voice={"wpm": 205, "barge_in": True, "latency_ms": 600}))
        self.assertGreater(p.value("tempo"), 0.65)
        self.assertLess(p.value("detail"), 0.5)
        self.assertEqual(round(p.voice_stats["wpm"]), 205)
        q = Persona()
        for _ in range(3):
            q.observe(utter("um, uh, maybe 2010 was, um, hot", "voice", voice={"wpm": 95, "latency_ms": 6000}))
        self.assertLess(q.value("tempo"), 0.4)
        self.assertLess(q.value("certainty"), 0.4)

    def test_perspective_is_what_they_talk_about(self):
        p = Persona()
        for t in ["she was injured", "her knee again", "it was hot", "why is the range so wide"]:
            p.observe(utter(t))
        focus = p.summary()["focus"]
        self.assertEqual(focus[0]["key"], "health")
        self.assertEqual({f["key"] for f in focus}, {"health", "conditions", "data"})

    def test_turn_taking_follows_the_speaker(self):
        e = AdaptiveEngine(Repository(":memory:"), "c-t")
        e.start(dict(STATE), "unknown")
        e.process(Event("voice_on"))
        for _ in range(4):
            r = e.process(utter("2010 hot", "voice", voice={"wpm": 210, "barge_in": True}))
        self.assertEqual(r["ui"]["voice_style"]["preset"], "quick")
        self.assertLess(r["ui"]["voice_style"]["silence_ms"], 1000)
        r = e.edit_persona({"voice_preset": "calm"})
        self.assertEqual(r["ui"]["voice_style"]["preset"], "calm")
        self.assertTrue(r["ui"]["voice_style"]["read_choices"])
        with self.assertRaises(ValueError):
            e.edit_persona({"voice_preset": "shouty"})

    def test_questions_are_spoken_with_how_to_answer(self):
        e = AdaptiveEngine(Repository(":memory:"), "c-s")
        e.start(dict(STATE), "unknown")
        self.assertTrue(e.speak(None).endswith("Say yes or no."))

    def test_explanations_use_the_session_numbers(self):
        self.assertIn("18 minutes slower", explain("race", STATE, 1))
        self.assertIn("20th to 80th percentiles", explain("range", STATE))


class VoiceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        cls.c = Client(create_app(StudioService(coach, settings=Settings())))

    def test_a_hands_free_session(self):
        r = self.c.post("/api/v1/sessions", {"coach_id": "coach-voice", "condition": "ask"}).json()
        sid = r["id"]
        self.assertIn("How well do you know this runner?", r["state"]["adapt"]["speak"])
        v = {"wpm": 160, "latency_ms": 700, "confidence": 0.9}
        r = self.c.post(f"/api/v1/sessions/{sid}/commands", {"text": "not at all", "source": "voice", "voice": v}).json()
        self.assertEqual(r["choice"]["loop"], "interview")
        self.assertEqual(r["adapt"]["ui"]["condition"], "unknown")
        self.assertEqual(r["adapt"]["prompt"]["loop"], "suggest")
        self.assertTrue(r["adapt"]["speak"].endswith("Say yes or no."))
        r = self.c.post(f"/api/v1/sessions/{sid}/commands", {"text": "why?", "source": "voice"}).json()
        self.assertIn("minutes", r["reply"])
        self.assertIsNotNone(r["adapt"]["prompt"])                            # the question is still open
        before = [x["weight"] for x in r["state"]["races"]]
        r = self.c.post(f"/api/v1/sessions/{sid}/commands", {"text": "yes", "source": "voice"}).json()
        after = [x["weight"] for x in r["state"]["races"]]
        self.assertNotEqual(before, after)                                   # "yes" moved a slider
        self.assertEqual(r["card"]["items"][0]["new"], 0.2)
        self.assertIsNotNone(r["card"]["items"][0]["index"])
        r = self.c.post(f"/api/v1/sessions/{sid}/commands", {"text": "yes", "source": "voice"}).json()
        r = self.c.post(f"/api/v1/sessions/{sid}/commands", {"text": "yes", "source": "voice"}).json()
        self.assertIn("no question open", r["reply"])                         # nothing left to say yes to

    def test_tapping_takes_the_same_path(self):
        sid = self.c.post("/api/v1/sessions", {"coach_id": "coach-tap", "condition": "unknown"}).json()["id"]
        r = self.c.post(f"/api/v1/sessions/{sid}/choice", {"index": 0}).json()
        self.assertTrue(r["card"]["items"])
        self.assertEqual(r["choice"]["loop"], "suggest")
        self.assertEqual(self.c.post(f"/api/v1/sessions/{sid}/choice", {"index": 9}).status, 422)

    def test_condition_route_and_validation(self):
        sid = self.c.post("/api/v1/sessions", {"coach_id": "coach-cond", "condition": "known"}).json()["id"]
        r = self.c.put(f"/api/v1/sessions/{sid}/condition", {"condition": "unknown"}).json()
        self.assertEqual(r["adapt"]["ui"]["condition"], "unknown")
        self.assertEqual(self.c.put(f"/api/v1/sessions/{sid}/condition", {"condition": "both"}).status, 422)
        self.assertEqual(self.c.post("/api/v1/sessions", {"condition": "maybe"}).status, 422)
        bad = {"text": "hi", "source": "voice", "voice": {"wpm": -3}}
        self.assertEqual(self.c.post(f"/api/v1/sessions/{sid}/commands", bad).status, 422)


if __name__ == "__main__":
    unittest.main()
