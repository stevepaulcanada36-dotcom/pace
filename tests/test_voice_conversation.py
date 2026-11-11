"""A spoken session is a conversation: questions are answered with the history, statements about a named
race move weights, and nothing the coach merely asks about changes the prediction."""
import json
import unittest

from pacesense.assistant.chat import BosonModel
from pacesense.coach import CoachService
from pacesense.hud.commands import is_question, repair_mishearings
from pacesense.llm.openrouter import ChatResult, Usage
from pacesense.storage import Repository
from pacesense.studio import Settings, StudioService
from tests import helpers


class ScriptedClient:
    """Stands in for the hosted model and records which of the three prompts each call used."""

    def __init__(self, intent=None):
        self.calls = []
        self.intent = intent or {"intent": "none", "arguments": {}, "confidence": 0.3, "evidence": [],
                                 "clarification": "The phrase is unclear. Could you clarify?"}

    def chat(self, model, messages, **kwargs):
        system = messages[0]["content"]
        kind = ("classifier" if system.startswith("You interpret one spoken command") else
                "steer" if "adjust a marathon prediction model by voice" in system else "assistant")
        self.calls.append((kind, messages))
        text = (json.dumps(self.intent) if kind == "classifier" else
                json.dumps({"actions": [], "estimate": None}) if kind == "steer" else "the assistant's answer")
        return ChatResult(text, [], Usage(), 0.0, model)

    def kinds(self):
        return [kind for kind, _ in self.calls]


class ScriptedAI:
    online = True
    provider = "openrouter"

    def __init__(self, client):
        self.client = client

    def config(self, task, model=None):
        return "openai/gpt-4o-mini", 0.0

    def chat_model(self, model=None):
        return BosonModel(self.client, "openai/gpt-4o-mini", 0.2, provider="OpenRouter")


class VoiceConversationTests(unittest.TestCase):
    def open(self, intent=None):
        client = ScriptedClient(intent)
        coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        service = StudioService(coach, settings=Settings(), ai_factory=lambda: ScriptedAI(client))
        sid = service.create()[0]
        live = service.sessions.entry(sid).value
        return service, sid, live, client

    @staticmethod
    def weights(live):
        return [round(r["weight"], 2) for r in live.hud.steer.state()["races"]]

    def test_a_question_about_a_race_is_answered_and_changes_nothing(self):
        service, sid, live, client = self.open()
        race = live.hud.steer.races[-1]
        before = self.weights(live)
        for said in (f"one {race['year']} what happened on that time",
                     f"you know like in the {race['year']} here what was what happened",
                     (f"in {race['year']} it's given like finished around 5 hours but you are talking about "
                     f"four 59 so which raise you're talking about is it not the one {race['year']}")):
            out = service.command(sid, said, source="voice")
            self.assertEqual(out["used"], ["assistant"], said)
            self.assertEqual(out["reply"], "the assistant's answer")
        self.assertEqual(self.weights(live), before)
        self.assertEqual(set(client.kinds()), {"assistant"})            # no classifier dead end, no steering

    def test_the_assistant_gets_the_runner_the_races_and_the_earlier_turns(self):
        service, sid, live, client = self.open()
        race = live.hud.steer.races[-1]
        service.command(sid, "so I just want to know what happened on", source="voice")
        service.command(sid, f"{race['year']} race", source="voice")
        _, messages = client.calls[-1]
        system = messages[0]["content"]
        self.assertIn(f"{race['event']} {race['year']}: finished in", system)
        self.assertIn("for the next marathon", system)
        self.assertIn("always mean the active runner", system)
        earlier = [m["content"] for m in messages[1:-1] if m["role"] == "user"]
        self.assertIn("so I just want to know what happened on", earlier)
        self.assertEqual(messages[-1]["content"], f"{race['year']} race")

    def test_a_statement_about_a_named_race_still_moves_its_weight(self):
        service, sid, live, _client = self.open()
        race = live.hud.steer.races[-1]
        before = self.weights(live)
        out = service.command(sid, f"{race['event']} {race['year']} was a hot day and he was injured", source="voice")
        self.assertIn("steer", out["used"])
        self.assertLess(self.weights(live)[-1], before[-1])

    def test_a_misheard_race_does_not_raise_a_weight(self):
        service, sid, live, _client = self.open()
        race = live.hud.steer.races[-1]
        before = self.weights(live)
        service.command(sid, f"they raise I'm talking about the race which happened on {race['year']}", source="voice")
        self.assertEqual(self.weights(live), before)

    def test_a_classifier_that_finds_no_control_hands_over_to_the_conversation(self):
        service, sid, _live, client = self.open()
        out = service.command(sid, "could you please show me all the details and insights", source="voice")
        self.assertEqual(client.kinds(), ["classifier", "assistant"])
        self.assertEqual(out["reply"], "the assistant's answer")

    def test_the_answer_is_spoken_not_the_question_that_was_already_open(self):
        service, sid, live, _client = self.open()
        self.assertEqual(live.engine.pending["message"], "How well do you know this runner?")
        for said in ("what does the likely range mean", "and what happened in the last race"):
            spoken = service.command(sid, said, source="voice")["adapt"]["speak"]
            self.assertTrue(spoken.startswith("the assistant's answer"), spoken)
        self.assertNotIn("How well do you know", spoken)

    def test_small_talk_costs_no_model_call(self):
        service, sid, _live, client = self.open()
        self.assertIn("listening", service.command(sid, "wait", source="voice")["reply"])
        self.assertIn("here", service.command(sid, "hello", source="voice")["reply"])
        self.assertEqual(client.calls, [])

    def test_yes_carries_out_the_panel_pacesense_offered(self):
        service, sid, live, _client = self.open({
            "intent": "focus_panel", "arguments": {"panel": "prediction"}, "confidence": 0.9,
            "evidence": ["prediction dashboard"], "clarification": ""})
        asked = service.command(sid, "I would like the prediction dashboard", source="voice")
        self.assertEqual(asked["reply"], "Do you want me to open the prediction panel?")
        self.assertEqual(asked["actions"], [])
        done = service.command(sid, "yep", source="voice")
        self.assertEqual(done["actions"], [{"type": "focus", "panel": "prediction"}])
        self.assertIsNone(live.voice_offer)
        again = service.command(sid, "I would like the prediction dashboard", source="voice")
        self.assertEqual(again["actions"], [])
        declined = service.command(sid, "no", source="voice")
        self.assertEqual(declined["actions"], [])
        self.assertIn("left the screen", declined["reply"])

    def test_an_injury_now_is_kept_as_context_not_pinned_on_an_old_race(self):
        service, sid, live, client = self.open()
        before = self.weights(live)
        said = "remember that runner had an injury to his back two days back so what will happen right now"
        out = service.command(sid, said, source="voice")
        self.assertEqual(self.weights(live), before)
        messages = [a.get("message", "") for a in out["adapt"].get("adaptations", [])]
        self.assertTrue(any("how the runner is now" in m for m in messages), messages)
        self.assertFalse(any(m.startswith("Which race was") for m in messages), messages)
        notes = live.engine.ctx.profile["runner_notes"][live.hud.source_runner]
        self.assertEqual(notes[-1]["text"], said)
        service.command(sid, "so what should I expect", source="voice")
        self.assertIn("Previously captured coach context", client.calls[-1][1][0]["content"])


class QuestionDetectionTests(unittest.TestCase):
    def test_questions(self):
        for said in ("what happened in 2019", "so I just want to know what happened on", "is it perfect",
                     "which race are you talking about", "why is the target faster than 2019",
                     "I'm asking you to predict the race at the end", "tell me about Amsterdam 2018",
                     "in 2019 it finished around five hours is it not the one in 2019"):
            self.assertTrue(is_question(said), said)

    def test_statements_and_instructions(self):
        for said in ("2019 was a hot day and she was injured", "trust the recent races more",
                     "I think she can run 4:43", "Amsterdam 2018 was her best race",
                     "why don't you count 2019 less", "reduce the weight of 2019"):
            self.assertFalse(is_question(said), said)

    def test_raise_is_read_as_race_only_where_a_race_is_meant(self):
        self.assertEqual(repair_mishearings("which raise you're talking about"), "which race you're talking about")
        self.assertEqual(repair_mishearings("the raise in 2019"), "the race in 2019")
        self.assertEqual(repair_mishearings("raise 2019 a bit"), "raise 2019 a bit")
        self.assertEqual(repair_mishearings("raise the weight of 2019"), "raise the weight of 2019")


if __name__ == "__main__":
    unittest.main()
