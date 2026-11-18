"""PaceSense as a fair second view: a counterweight on every change, and a grounded second opinion."""
import json
import unittest

from pacesense.hud.critique import consider, effective_races
from pacesense.hud.second_opinion import SECOND_OPINION_RE, grounded, second_opinion
from pacesense.llm.openrouter import ChatResult, Usage
from tests.test_guide import open_session


def settled(service, sid, live):
    service.command(sid, "not at all", source="typed")
    while live.engine.pending:
        service.command(sid, "no", source="typed")


class CounterweightTests(unittest.TestCase):
    def test_a_hope_that_concentrates_the_target_on_one_race_is_challenged(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        races = live.hud.steer.races
        for r in races[:-1]:
            service.command(sid, f"set {r['event']} {r['year']} to 0.0", source="typed")
        out = service.command(sid, f"maybe {races[-1]['year']} is the most representative race", source="typed")
        kinds = {p["kind"]: p["text"] for p in out["card"]["consider"]}
        self.assertIn("against", kinds)
        texts = " ".join(p["text"] for p in out["card"]["consider"])
        self.assertIn("rests mostly on", texts)
        self.assertIn("One thing to weigh", out["reply"])

    def test_a_concrete_reason_is_credited(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        r = live.hud.steer.races[0]
        out = service.command(sid, f"she was injured in {r['year']}", source="typed")
        self.assertIn("for", [p["kind"] for p in out["card"]["consider"]])

    def test_effective_races(self):
        self.assertAlmostEqual(effective_races([0.5, 0.5, 0.5, 0.5]), 4.0)
        self.assertLess(effective_races([1.0, 0.0, 0.0, 0.1]), 1.3)

    def test_no_change_no_counterweight(self):
        self.assertEqual(consider({"items": []}, {}), [])


class FakeAI:
    online = True

    def __init__(self, payload):
        self.payload, self.calls = payload, []
        self.client = self

    def config(self, task, model=None):
        return "openai/gpt-4o-mini", 0.2

    def chat(self, model, messages, **kw):
        self.calls.append(messages)
        return ChatResult(json.dumps(self.payload), [], Usage(), 0.0, model)


class SecondOpinionTests(unittest.TestCase):
    def test_requests_are_recognised(self):
        for said in ("what's the case against my changes?", "is the target overfitting?", "give me a second opinion",
                     "am I being biased here", "do you agree with me", "play devil's advocate"):
            self.assertTrue(SECOND_OPINION_RE.search(said), said)
        self.assertFalse(SECOND_OPINION_RE.search("2019 was a hot day"))

    def test_a_sentence_with_an_invented_number_is_dropped(self):
        allowed = {"3:35:12", "3:35", "0.9", "26"}
        self.assertTrue(grounded("The target is 3:35 now.", allowed))
        self.assertTrue(grounded("It rests on one race.", allowed))
        self.assertFalse(grounded("She will run 3:20.", allowed))
        self.assertFalse(grounded("That is 85% likely.", allowed))

    def test_model_opinion_is_filtered_and_falls_back_when_nothing_is_left(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        r = live.hud.steer.races[0]
        service.command(sid, f"set {r['event']} {r['year']} to 1.0", source="typed")
        state = service.view(sid, live)
        good = FakeAI({"verdict": "Your view rests on one race.", "for": ["You gave a reason."],
                       "against": ["She will run 2:59 for sure.", "The other races still disagree."],
                       "settle": "Reveal the hidden race."})
        op = second_opinion(good, state, said=["count it fully"], changes=live.changes, last_card=live.last_card,
                            analysis=state["session_analysis"], question="second opinion")
        self.assertEqual(op["source"], "model")
        self.assertEqual(op["against"], ["The other races still disagree."])          # the invented 2:59 is gone
        self.assertIn("Limits:", good.calls[0][0]["content"])
        bad = FakeAI({"verdict": "She will run 2:59.", "for": [], "against": ["90% sure."], "settle": ""})
        op = second_opinion(bad, state, said=[], changes=live.changes, last_card=live.last_card,
                            analysis=state["session_analysis"], question="second opinion")
        self.assertEqual(op["source"], "rules")

    def test_in_the_app_without_a_model(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        r = live.hud.steer.races[0]
        service.command(sid, f"set {r['event']} {r['year']} to 1.0", source="typed")
        out = service.command(sid, "what's the case against my changes?", source="typed")
        self.assertIn("opinion", out)
        self.assertTrue(out["opinion"]["verdict"])
        self.assertEqual(out["reply"].split(".")[0] + ".", out["opinion"]["verdict"])
        self.assertNotIn("second", [s["key"] for s in out["state"]["next_steps"]])


if __name__ == "__main__":
    unittest.main()
