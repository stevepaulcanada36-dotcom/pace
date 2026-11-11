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

    def test_stray_speech_is_not_answered_aloud_or_remembered(self):
        service, sid, live, client = self.open()
        for said in ("OK Google", "sorry babe", "you too", "movie", "show", "this relationship", "I think"):
            out = service.command(sid, said, source="voice")
            self.assertEqual(out["adapt"]["speak"], "", said)
            self.assertIn("didn't catch a request", out["reply"])
        self.assertEqual(client.calls, [])
        self.assertEqual(live.hud.history, [])
        race = live.hud.steer.races[-1]                      # a year or a race name is still a real turn
        service.command(sid, str(race["year"]), source="voice")
        self.assertEqual(client.kinds(), ["assistant"])

    def test_a_sentence_sharing_one_word_with_a_choice_is_not_an_answer(self):
        service, sid, live, _client = self.open()
        self.assertEqual(live.engine.pending["message"], "How well do you know this runner?")
        out = service.command(sid, "the 2019 data is not being much more accurate", source="voice")
        self.assertNotIn("choice", out)
        self.assertEqual(live.engine.ctx.session.condition, "ask")
        out = service.command(sid, "not at all", source="voice")
        self.assertEqual(out["choice"]["label"], "Not at all")

    def test_guide_me_starts_the_interview_by_voice(self):
        service, sid, live, client = self.open()
        service.command(sid, "not at all", source="voice")
        out = service.command(sid, "guide me", source="voice")
        self.assertEqual(client.calls, [])
        self.assertTrue(live.engine.ctx.session.interview_on)
        self.assertTrue(out["adapt"]["speak"].startswith("Starting the guided interview."))

    def test_an_injury_is_asked_about_once_and_not_during_a_race_closes_it(self):
        service, sid, live, _client = self.open()
        service.command(sid, "not at all", source="voice")
        before = self.weights(live)
        service.command(sid, "she had an injury in her leg", source="voice")
        self.assertEqual(live.engine.pending["message"], 'Which race was "injury"?')
        out = service.command(sid, "no it didn't happen during the race it happened yesterday", source="voice")
        self.assertIn("how the runner is now", out["reply"])
        self.assertNotEqual((live.engine.pending or {}).get("loop"), "clarify")
        service.command(sid, "for the injury what should I do about it", source="voice")
        self.assertNotEqual((live.engine.pending or {}).get("loop"), "clarify")      # not asked a second time
        self.assertEqual(self.weights(live), before)
        notes = live.engine.ctx.profile["runner_notes"][live.hud.source_runner]
        self.assertTrue(any("yesterday" in n["text"] for n in notes))

    def test_yes_after_an_offer_from_the_assistant_goes_back_to_the_assistant(self):
        service, sid, live, client = self.open()
        service.command(sid, "not at all", source="voice")
        service.command(sid, "what can I do about her training this month", source="voice")
        live.hud.history[-1]["content"] = "Would you like me to run a simulation for a hot day?"
        n = len(client.calls)
        out = service.command(sid, "yes go ahead", source="voice")
        self.assertEqual(out["reply"], "the assistant's answer")
        self.assertEqual(len(client.calls), n + 1)

    def test_the_open_question_is_not_copied_into_every_turn_of_the_history(self):
        service, sid, live, _client = self.open()
        for said in ("what does the likely range mean", "and how sure is the model about that"):
            service.command(sid, said, source="voice")
        self.assertFalse(any("I coach her" in m["content"] for m in live.hud.history))

    def test_saying_a_race_is_not_precise_counts_it_less(self):
        service, sid, live, _client = self.open()
        service.command(sid, "not at all", source="voice")
        race = live.hud.steer.races[1]
        before = self.weights(live)
        out = service.command(sid, f"{race['year']} datas are not precise so pretty picture is it", source="voice")
        self.assertIn("steer", out["used"])
        self.assertLess(self.weights(live)[1], before[1])

    def test_an_odd_speech_measurement_does_not_reject_the_turn(self):
        from pacesense.studio.schemas import Command
        body = Command.model_validate({"text": "hello", "source": "voice",
                                       "voice": {"wpm": 1800, "duration_ms": 9e9, "confidence": 1.4}})
        self.assertEqual((body.voice.wpm, body.voice.duration_ms, body.voice.confidence), (600, 120000, 1))

    def test_describing_how_well_you_know_the_runner_answers_the_open_question(self):
        for said, label in (("Like I'm not ever much of this runner, but still", "Not at all"),
                            ("I've seen him once, maybe in 2013, like a long time", "A little"),
                            ("I coach her and have done for years", "I coach her"),
                            ("I don't know this runner", "Not at all")):
            service, sid, _live, _client = self.open()
            out = service.command(sid, said, source="voice")
            self.assertEqual(out.get("choice", {}).get("label"), label, said)

    def test_yes_with_nothing_asked_is_not_sent_to_the_model(self):
        service, sid, _live, client = self.open()
        service.command(sid, "not at all", source="voice")
        service.command(sid, "no", source="voice")                       # closes the suggestion, if any
        service.command(sid, "what does the likely range mean", source="voice")
        n = len(client.calls)
        out = service.command(sid, "Yeah.", source="voice")
        self.assertEqual(len(client.calls), n)
        self.assertIn("no question open", out["reply"])

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


class ToolCallingClient(ScriptedClient):
    """A model that answers the assistant prompt by calling one function, then claims success in words."""

    def __init__(self, name, arguments, claim="I've set that for you. The prediction is now 3:00:00."):
        super().__init__()
        self.name, self.arguments, self.claim, self.results = name, arguments, claim, []

    def chat(self, model, messages, **kwargs):
        system = messages[0]["content"]
        if system.startswith("You interpret one spoken command") or "adjust a marathon prediction model" in system:
            return super().chat(model, messages, **kwargs)
        self.calls.append(("assistant", messages))
        tool_results = [m for m in messages if m.get("role") == "tool"]
        if self.name and not tool_results:
            return ChatResult("", [{"id": "call_1", "name": self.name, "arguments": self.arguments}],
                              Usage(), 0.0, model)
        self.results = [json.loads(m["content"]) for m in tool_results]
        return ChatResult(self.claim, [], Usage(), 0.0, model)


class AssistantActionTests(unittest.TestCase):
    def open(self, name, arguments, **kw):
        client = ToolCallingClient(name, arguments, **kw)
        coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        service = StudioService(coach, settings=Settings(), ai_factory=lambda: ScriptedAI(client))
        sid = service.create()[0]
        live = service.sessions.entry(sid).value
        service.command(sid, "not at all", source="voice")
        service.command(sid, "no", source="voice")
        return service, sid, live, client

    def test_the_assistant_can_really_change_a_weight(self):
        service, sid, live, client = self.open("set_race_weight", {"race": "PLACEHOLDER", "weight": 0.3})
        race = live.hud.steer.races[-1]
        client.arguments = {"race": f"{race['event']} {race['year']}", "weight": 0.3}
        out = service.command(sid, "I hope you can change the last one because he should go under five hours",
                              source="voice")
        self.assertEqual(round(live.hud.steer.state()["races"][-1]["weight"], 2), 0.3)
        self.assertTrue(client.results[0]["applied"])
        self.assertIn("steer", out["used"])
        self.assertIsNotNone(out["card"])
        self.assertNotIn("3:00:00", out["reply"])             # the wording comes from PaceSense, not the model
        self.assertTrue(live.hud.steer._undo)

    def test_the_assistant_can_record_the_coachs_estimate(self):
        service, sid, live, _client = self.open("set_coach_estimate", {"finish": "4:30"})
        service.command(sid, "I believe the person may complete in about four hours thirty minutes", source="voice")
        self.assertEqual(live.hud.state()["steering"]["coach_estimate_s"], 4.5 * 3600)

    def test_a_claimed_change_without_a_function_call_is_corrected(self):
        service, sid, live, _client = self.open(None, None, claim="I've set the weight for the last race to 0.30.")
        before = [r["weight"] for r in live.hud.steer.state()["races"]]
        out = service.command(sid, "yep you can proceed with that one for me", source="voice")
        self.assertEqual([r["weight"] for r in live.hud.steer.state()["races"]], before)
        self.assertTrue(out["reply"].startswith("Nothing has changed yet."))

    def test_an_unknown_race_is_refused_with_the_list(self):
        service, sid, live, client = self.open("set_race_weight", {"race": "Berlin 1999", "weight": 0.3},
                                               claim="I could not find that race.")
        before = [r["weight"] for r in live.hud.steer.state()["races"]]
        service.command(sid, "please change the Berlin one for me now", source="voice")
        self.assertFalse(client.results[0]["applied"])
        self.assertEqual(len(client.results[0]["races"]), len(before))
        self.assertEqual([r["weight"] for r in live.hud.steer.state()["races"]], before)

    def test_the_revealed_race_is_explained_to_the_assistant(self):
        service, sid, _live, client = self.open(None, None, claim="It is the hidden race.")
        service.command(sid, "what is the accuracy check about", source="voice")
        self.assertIn("is hidden from the model", client.calls[-1][1][0]["content"])
        service.reveal(sid)
        service.command(sid, "so how are we predicting that year", source="voice")
        system = client.calls[-1][1][0]["content"]
        self.assertIn("has now been revealed", system)
        self.assertIn("% away from that result", system)


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
