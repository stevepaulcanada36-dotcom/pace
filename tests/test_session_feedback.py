"""From a live session: words that are not about a race never move a weight; questions about a future race,
"I'm clueless" and panel requests get grounded answers."""
import unittest

from pacesense.coach import CoachService
from pacesense.hud.commands import refers_to_race
from pacesense.storage import Repository
from pacesense.studio import Settings, StudioService
from tests import helpers
from tests.test_critique import settled
from tests.test_guide import open_session
from tests.test_voice_conversation import ScriptedAI, ToolCallingClient


def weights(live):
    return [round(r["weight"], 2) for r in live.hud.steer.state()["races"]]


class NoStrayChangesTests(unittest.TestCase):
    def test_sentences_about_no_race_change_nothing(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        r = live.hud.steer.races[-1]
        service.command(sid, f"in {r['year']} it looks like it was slower", source="voice")
        before = weights(live)
        for said in ("And I'm going to predict for the 2026 marathon. This is, yeah, this is totally meaningless.",
                     "No, like it's totally meaningless because the accuracy is odd.",
                     "I am asking 2026 marathon. Marathon."):
            service.command(sid, said, source="voice")
            self.assertEqual(weights(live), before, said)

    def test_race_references(self):
        races = [{"event": "Amsterdam", "year": 2013}, {"event": "Eindhoven", "year": 2019}]
        self.assertTrue(refers_to_race("she was injured in 2013", races))
        self.assertTrue(refers_to_race("trust the recent races more", races))
        self.assertTrue(refers_to_race("I think she can run 4:43", races))
        self.assertFalse(refers_to_race("this is totally meaningless", races))
        self.assertFalse(refers_to_race("I am asking about the 2026 marathon", races))


class GroundedAnswersTests(unittest.TestCase):
    def test_a_2026_question_gets_the_target_and_what_the_accuracy_check_is(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        out = service.command(sid, "How much time will it take for her to complete the 2026 marathon?", source="typed")
        self.assertIn("next marathon", out["reply"])
        self.assertIn("best past race", out["reply"])
        self.assertIn(out["state"]["prediction"]["finish"][:4], out["reply"])
        self.assertEqual(out["actions"], [{"type": "focus", "panel": "prediction"}])
        tomorrow = service.command(sid, "the upcoming race is tomorrow, how will she perform?", source="typed")
        self.assertIn("next marathon", tomorrow["reply"])

    def test_clueless_gets_the_picture_and_one_thing_to_try(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        for said in ("I'm clueless. So, what can I do now?", "you suggest me what to do"):
            out = service.command(sid, said, source="voice")
            self.assertIn("Her target is", out["reply"])
            self.assertIn("Try saying", out["reply"])
            self.assertIn("moves it most", out["reply"])
        self.assertNotIn("new runner", out["reply"])

    def test_open_the_contribution_panel(self):
        service, sid, live = open_session()
        settled(service, sid, live)
        for said in ("Okay. Open it. Open the contribution panel.",
                     "To show the raise contribution, to access the contribution panel."):
            out = service.command(sid, said, source="voice")
            self.assertEqual(out["actions"], [{"type": "focus", "panel": "contributions"}], said)

    def test_the_assistant_can_open_a_panel_itself(self):
        client = ToolCallingClient("open_panel", {"panel": "similar"}, claim="Here are similar runners.")
        service = StudioService(CoachService(helpers.model(), helpers.runners(), Repository(":memory:")),
                                settings=Settings(), ai_factory=lambda: ScriptedAI(client))
        sid = service.create()[0]
        live = service.sessions.entry(sid).value
        settled(service, sid, live)
        out = service.command(sid, "I would like to compare her with runners like her", source="voice")
        self.assertIn({"type": "focus", "panel": "similar"}, out["actions"])

    def test_the_assistant_is_told_the_hidden_race_is_a_past_race(self):
        client = ToolCallingClient(None, None, claim="ok")
        service = StudioService(CoachService(helpers.model(), helpers.runners(), Repository(":memory:")),
                                settings=Settings(), ai_factory=lambda: ScriptedAI(client))
        sid = service.create()[0]
        live = service.sessions.entry(sid).value
        settled(service, sid, live)
        service.command(sid, "what does the likely range mean", source="voice")
        system = client.calls[-1][1][0]["content"]
        self.assertIn("best past race", system)
        self.assertIn("it is not the next race", system)
        self.assertNotIn("real next marathon", system)


if __name__ == "__main__":
    unittest.main()
