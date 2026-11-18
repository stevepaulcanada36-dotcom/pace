"""The reasoning loop: claims are recorded, tested against the hidden race, and reflected on by the model with every
number checked."""
import unittest

from pacesense.hud import loop
from tests.test_critique import FakeAI, settled
from tests.test_guide import open_session

REFLECTION = {"think": ["The coach says that race was slow for a reason the times cannot show.",
                        "Counting it less moves the target, and she will run 2:41 for sure.",
                        "The hidden race will test it."],
              "say": "That fits what you said about the heat. She will surely run 2:41 next time.",
              "ask": "Was anything else unusual about that day?",
              "stance": "partly"}


def change(service, sid, live, weight=0.0, i=0):
    r = live.hud.steer.races[i]
    return service.command(sid, f"set {r['event']} {r['year']} to {weight}", source="typed")


class LedgerTests(unittest.TestCase):
    def test_each_change_becomes_a_claim_and_a_new_change_supersedes_it(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        change(service, sid, live, 0.0)
        change(service, sid, live, 0.9)
        live_claims = loop.active(live.claims)
        self.assertEqual(len(live_claims), 1)
        self.assertEqual(live_claims[0]["direction"], "more")
        self.assertEqual(len(service.view(sid, live)["claims"]), 1)

    def test_reveal_tests_each_claim_against_the_counterfactual(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        change(service, sid, live, 0.0, 0)
        change(service, sid, live, 1.0, 1)
        st = service.reveal(sid)
        statuses = {c["race"]: c for c in st["claims"]}
        self.assertEqual(len(statuses), 2)
        truth = live.hud.truth_s
        races = st["races"]
        weights = [r["weight"] for r in races]
        for c in st["claims"]:
            self.assertIn(c["status"], ("helped", "hurt", "neutral"))
            i = next(k for k, r in enumerate(races) if f"{r['event']} {r['year']}" == c["race"])
            without = list(weights)
            without[i] = 0.5
            now = sum(w * r["prediction_s"] for w, r in zip(weights, races)) / sum(weights)
            alt = sum(w * r["prediction_s"] for w, r in zip(without, races)) / sum(without)
            expected = abs(alt - truth) - abs(now - truth)
            self.assertEqual(c["status"] == "helped", expected / truth * 100 > 0.01, c)
        self.assertTrue(st["loop_reply"])                     # the closure is said even without a model

    def test_the_target_formula_matches_the_model(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        change(service, sid, live, 0.1, 0)
        st = service.view(sid, live)
        races = st["races"]
        self.assertAlmostEqual(loop._target(races, [r["weight"] for r in races]), st["prediction"]["finish_s"], delta=1)


class ReflectionTests(unittest.TestCase):
    def test_a_change_gets_the_models_reflection_with_invented_numbers_removed(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        live.hud.ai = ai = FakeAI(REFLECTION)
        out = change(service, sid, live, 0.0)
        self.assertEqual(out["reply"], "That fits what you said about the heat. Was anything else unusual about that day?")
        self.assertNotIn("2:41", " ".join(out["card"]["thinking"]))
        self.assertEqual(len(out["card"]["thinking"]), 2)
        self.assertNotIn("next_say", out)                       # one question per turn
        sheet = ai.calls[-1][0]["content"]
        self.assertIn("Claim 1", sheet)
        self.assertIn("What just happened: the coach changed race weights", sheet)

    def test_later_turns_see_earlier_claims_and_earlier_replies(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        live.hud.ai = ai = FakeAI(REFLECTION)
        change(service, sid, live, 0.0, 0)
        change(service, sid, live, 1.0, 1)
        messages = ai.calls[-1]
        self.assertIn("Claim 2", messages[0]["content"])
        self.assertIn("Claim 1", messages[0]["content"])
        self.assertIn("do not repeat", messages[1]["content"])

    def test_the_slider_and_the_reveal_are_reflected_on_too(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        live.hud.ai = FakeAI(REFLECTION)
        out = service.set_weight(sid, 0, 0.2)
        self.assertIn("reflection", out)
        self.assertTrue(out["card"]["thinking"])
        st = service.reveal(sid)
        self.assertEqual(st["loop_reply"], "That fits what you said about the heat. Was anything else unusual about that day?")

    def test_a_reflection_without_a_grounded_reply_falls_back(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        live.hud.ai = FakeAI({"think": ["x"], "say": "She will run 2:41.", "ask": "", "stance": "agree"})
        out = change(service, sid, live, 0.0)
        self.assertNotIn("reflection", out)
        self.assertIn("→", out["reply"])                         # PaceSense's own reply

    def test_offline_nothing_changes(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        out = change(service, sid, live, 0.0)
        self.assertNotIn("reflection", out)
        self.assertNotIn("thinking", out["card"])

    def test_grounding_accepts_numbers_on_the_sheet(self):
        allowed = loop._allowed("Target 3:35:12; 2.33% off; weight 0.9")
        self.assertTrue(loop.grounded("Now 3:35, 2.33 points.", allowed))
        self.assertFalse(loop.grounded("Maybe 3:20.", allowed))


if __name__ == "__main__":
    unittest.main()


class LiveSessionFixesTests(unittest.TestCase):
    """From a live session on 18 Nov."""

    def test_a_change_made_by_the_assistant_is_reflected_on_too(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        live.hud.ai = FakeAI(REFLECTION)
        from pacesense.hud.commands import Routed

        r = live.hud.steer.races[0]
        turn = live.hud.steer.manual(0, 0.8).to_dict()
        import pacesense.studio.service as svc
        original = svc.route
        svc.route = lambda text, s, source: Routed(reply="template", turn={**turn, "said": text},
                                                   actions=[{"type": "refresh"}], used=["assistant", "steer"])
        try:
            out = service.command(sid, f"make {r['year']} a bit less", source="typed")
        finally:
            svc.route = original
        self.assertIn("reflection", out)
        self.assertTrue(out["card"]["thinking"])

    def test_spoken_times(self):
        from pacesense.hud.commands import spoken_times
        self.assertEqual(spoken_times("I hope she completes it by three twenty minutes"), "I hope she completes it by 3:20")
        self.assertEqual(spoken_times("the prediction will be like three twenty one"), "the prediction will be like 3:21")
        self.assertEqual(spoken_times("close to 350"), "close to 3:50")
        self.assertEqual(spoken_times("in 2013 she ran well"), "in 2013 she ran well")

    def test_seeing_a_race_is_a_reason_not_a_guess(self):
        from pacesense.hud.critique import FACT_RE
        self.assertTrue(FACT_RE.search("I know I've seen this race in 2013"))

    def test_side_conversation_is_named_once_then_waited_out(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        import pacesense.studio.service as svc
        from pacesense.hud.commands import Routed
        original = svc.route
        svc.route = lambda text, s, source: Routed(reply="I didn't catch a question about the runner there.",
                                                   used=["assistant"])
        try:
            first = service.command(sid, "how was the quiz today, were you able to answer it", source="voice")
            second = service.command(sid, "we still have two more classes this term right", source="voice")
        finally:
            svc.route = original
        self.assertIn("talking to someone else", first["reply"])
        self.assertEqual(second["adapt"]["speak"], "")
        self.assertTrue(second.get("unheard"))
        self.assertFalse(any("quiz" in str(m.get("content")) for m in live.hud.history))

    def test_a_long_sentence_answers_the_familiarity_question_and_still_counts(self):
        service, sid, live = open_session()
        pend = live.engine.pending
        self.assertTrue(pend)
        r = live.hud.steer.races[0]
        out = service.command(sid, "So, I think I'm not much aware of this runner, like, I don't know her at all. "
                                   f"But I know I've seen the {r['event']} race, which is in {r['year']}, like she was "
                                   "pretty much good in this race, I would count it a lot more honestly.",
                              source="typed")
        self.assertTrue(out["reply"].startswith("Noted"))
        self.assertEqual(live.engine.ctx.session.condition, "unknown")
