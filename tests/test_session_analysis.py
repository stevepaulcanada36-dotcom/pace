"""The paper's measures applied to one session (Acceptance, Model adaptation, accuracy, recency, knowledge)."""
import unittest

from pacesense.hud.session_analysis import session_analysis, summary
from tests.test_guide import open_session

H = 3600


def state(model_initial, model_now, coach=None, races=None, holdout=None):
    return {"prediction": {"initial_s": model_initial, "finish_s": model_now},
            "steering": {"coach_estimate_s": coach}, "races": races or [], "holdout": holdout or {}}


class WhoseViewTests(unittest.TestCase):
    """The four cases from the paper's Acceptance discussion (slide 10 of the presentation)."""

    def case(self, model_initial, model_now, first, now):
        a = session_analysis(state(model_initial, model_now, now), coach_first_s=first, condition="unknown", said=[])
        return a["whose_view"]

    def test_coach_persuaded(self):
        wv = self.case(3.6 * H, 3.6 * H, 3.3 * H, 3.6 * H)
        self.assertEqual((wv["case"], wv["acceptance"]), (1, 1.0))

    def test_met_halfway(self):
        self.assertEqual(self.case(3.6 * H, 3.5 * H, 3.3 * H, 3.3 * H)["case"], 2)

    def test_model_steered_to_coach_still_scores_full_acceptance(self):
        wv = self.case(3.6 * H, 3.31 * H, 3.3 * H, 3.3 * H)
        self.assertEqual((wv["case"], wv["acceptance"]), (3, 1.0))
        self.assertIn("steering", wv["text"])

    def test_steered_beyond(self):
        self.assertEqual(self.case(3.6 * H, 3.1 * H, 3.3 * H, 3.3 * H)["case"], 4)

    def test_no_estimate_asks_for_one(self):
        a = session_analysis(state(3.6 * H, 3.6 * H), coach_first_s=None, condition="ask", said=[])
        self.assertFalse(a["whose_view"]["available"])
        self.assertIn("I think she can run", a["whose_view"]["text"])


class OtherMeasuresTests(unittest.TestCase):
    def test_accuracy_after_reveal_compares_model_and_coach(self):
        a = session_analysis(state(3.6 * H, 3.5 * H, 3.45 * H, holdout={"label": "Rotterdam 2015", "revealed": True,
                                                                        "truth_s": 3.5 * H}),
                             coach_first_s=3.45 * H, condition="unknown", said=[])
        rows = {r["name"]: r["pct"] for r in a["accuracy"]["rows"]}
        self.assertEqual(rows["Your model"], 0.0)
        self.assertGreater(rows["Equal weights"], 0)
        self.assertIn("more accurate", a["accuracy"]["text"])

    def test_recency_matches_the_papers_direction(self):
        races = [{"year": y, "weight": w} for y, w in ((2012, 0.3), (2014, 0.5), (2017, 0.9))]
        a = session_analysis(state(3 * H, 3 * H, races=races), coach_first_s=None, condition="unknown", said=[])
        self.assertGreater(a["recency"]["per_year"], 0)
        self.assertIn("like the paper's coaches", a["recency"]["text"])

    def test_knowledge_is_grouped_as_in_the_paper(self):
        said = ["2019 was a hot day", "she had an injury in her knee", "she has been training hard", "she is a good runner"]
        a = session_analysis(state(3 * H, 3 * H), coach_first_s=None, condition="unknown", said=said)
        names = {c["name"] for c in a["knowledge"]["categories"]}
        self.assertTrue({"Race-day conditions", "Injuries and health", "Training load", "Untapped potential"} <= names)

    def test_summary_leads_with_what_is_available(self):
        a = session_analysis(state(3 * H, 3 * H), coach_first_s=None, condition="unknown", said=[])
        self.assertTrue(summary(a))


class InTheAppTests(unittest.TestCase):
    def test_how_did_i_do_opens_the_session_and_answers_from_the_numbers(self):
        service, sid, live = open_session()
        service.command(sid, "not at all", source="typed")
        while live.engine.pending:
            service.command(sid, "no", source="typed")
        race = live.hud.steer.races[-1]
        service.command(sid, "I think she can run 3:30", source="typed")
        service.command(sid, f"set {race['event']} {race['year']} to 0.9", source="typed")
        service.reveal(sid)
        out = service.command(sid, "how did I do?", source="typed")
        self.assertEqual(out["actions"], [{"type": "focus", "panel": "session"}])
        self.assertIn("% off", out["reply"])
        analysis = out["state"]["session_analysis"]
        self.assertTrue(analysis["whose_view"]["available"])
        self.assertEqual(analysis["whose_view"]["points"]["coach_first_s"], 3.5 * H)


if __name__ == "__main__":
    unittest.main()
