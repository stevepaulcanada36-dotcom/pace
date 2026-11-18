"""Reasoning and next-step guidance: computed from PaceSense's numbers, and true to what the model does."""
import re
import unittest

from pacesense.coach import CoachService
from pacesense.domain import parse_time
from pacesense.hud.guide import next_steps, reasoning
from pacesense.storage import Repository
from pacesense.studio import Settings, StudioService
from tests import helpers


def open_session():
    service = StudioService(CoachService(helpers.model(), helpers.runners(), Repository(":memory:")),
                            settings=Settings())
    sid = service.create()[0]
    return service, sid, service.sessions.entry(sid).value


class ReasoningTests(unittest.TestCase):
    def test_the_stated_direction_matches_what_the_target_did_for_every_race(self):
        for _ in range(4):
            service, sid, live = open_session()
            for race in live.hud.steer.races:
                name = f"{race['event']} {race['year']}"
                for weight in (0.0, 0.9):
                    out = service.command(sid, f"set {name} to {weight}", source="typed")
                    if not out["card"]:
                        continue
                    delta = out["card"]["target_delta"]
                    said = next((line for line in out["card"]["reasoning"] if "moves the target" in line), "")
                    if delta and said and delta.strip("+-") not in ("0:00", "0:00:00"):
                        expected = "slower" if delta.startswith("+") else "faster"
                        self.assertIn(f"moves the target {expected}", said, (name, weight, delta, said))
                    service.command(sid, "undo", source="typed")

    def test_a_slow_race_can_still_pull_the_target_faster_and_the_reason_says_why(self):
        service, sid, live = open_session()
        state = service.view(sid, live)
        target = state["prediction"]["finish_s"]
        tricky = next((r for r in state["races"] if r["evidence"]["direction"] > 0 and r["prediction_s"] < target), None)
        if tricky is None:
            self.skipTest("this demo runner has no such race")
        out = service.command(sid, f"set {tricky['event']} {tricky['year']} to 0.0", source="typed")
        self.assertTrue(out["card"]["target_delta"].startswith("+"))
        self.assertIn("slower than the runner's other races", " ".join(out["card"]["reasoning"]))
        self.assertIn("moves the target slower", out["reply"])

    def test_no_reasoning_when_nothing_changed(self):
        self.assertEqual(reasoning({"items": []}, []), [])


class NextStepTests(unittest.TestCase):
    def test_steps_follow_what_the_coach_has_done(self):
        service, sid, live = open_session()
        first = [s["key"] for s in service.view(sid, live)["next_steps"]]
        self.assertEqual(first[0], "familiarity")
        service.command(sid, "not at all", source="typed")
        race = live.hud.steer.races[0]
        service.command(sid, f"set {race['event']} {race['year']} to 0.8", source="typed")
        keys = [s["key"] for s in service.view(sid, live)["next_steps"]]
        self.assertNotIn("familiarity", keys)
        self.assertTrue({"estimate", "reveal"} & set(keys), keys)
        service.reveal(sid)
        self.assertNotIn("reveal", [s["key"] for s in service.view(sid, live)["next_steps"]])

    def test_every_step_has_words_to_say_and_a_reason(self):
        service, sid, live = open_session()
        for step in service.view(sid, live)["next_steps"]:
            self.assertTrue(step["say"] and step["why"].endswith("."), step)

    def test_steps_rotate_instead_of_repeating(self):
        service, sid, live = open_session()
        state = service.view(sid, live)
        shown: dict = {}
        tops = []
        for _ in range(3):
            top = next_steps(state, familiarity="known", used=set(), shown=shown)[0]
            tops.append(top["key"])
            shown[top["key"]] = shown.get(top["key"], 0) + 1
        self.assertEqual(len(set(tops)), 3)                      # three different suggestions in a row

    def test_a_change_brings_a_next_step_spoken_only_the_first_time(self):
        service, sid, live = open_session()
        service.command(sid, "not at all", source="typed")
        while live.engine.pending:                                    # close any open question
            service.command(sid, "no", source="typed")
        races = live.hud.steer.races
        first = service.command(sid, f"set {races[0]['event']} {races[0]['year']} to 0.8", source="typed")
        self.assertIn("next_step", first)
        self.assertTrue(first.get("next_say", "").startswith("Next, you could say"))
        key = first["next_step"]["key"]
        again = service.command(sid, f"set {races[1]['event']} {races[1]['year']} to 0.8", source="typed")
        if again.get("next_step", {}).get("key") == key:
            self.assertNotIn("next_say", again)

    def test_the_estimate_suggestion_is_a_real_time(self):
        service, sid, live = open_session()
        race = live.hud.steer.races[0]
        service.command(sid, f"set {race['event']} {race['year']} to 0.8", source="typed")
        step = next(s for s in next_steps(service.view(sid, live), familiarity="unknown", used={"changed"}, shown={})
                    if s["key"] == "estimate")
        self.assertIsNotNone(parse_time(re.search(r"\d:\d{2}:\d{2}", step["say"]).group(0)))


if __name__ == "__main__":
    unittest.main()
