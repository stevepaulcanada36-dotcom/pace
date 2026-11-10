"""Validation tests for AI-interpreted voice actions."""
import json
import unittest
from types import SimpleNamespace

from pacesense.assistant.chat import ModelReply
from pacesense.coach import CoachService
from pacesense.hud.commands import ui_commands
from pacesense.hud.voice_actions import MIN_CONFIDENCE, validate_voice_intent
from pacesense.storage import Repository
from pacesense.studio import Settings, StudioService
from pacesense.studio.app import create_app
from pacesense.studio.schemas import Command
from tests import helpers
from tests.asgi import Client

RUNNERS = [
    {"runner_id": "R00042", "name": "Alex Smith"},
    {"runner_id": "R00043", "name": "Alex Smith"},
    {"runner_id": "R00044", "name": "Jamie Lee"},
]


def intent(name, arguments=None, confidence=0.95, evidence=None):
    return {"intent": name, "arguments": arguments or {}, "confidence": confidence,
            "evidence": evidence if evidence is not None else ["The coach explicitly requested this action."]}


class VoiceIntentValidationTests(unittest.TestCase):
    def test_low_confidence_requires_clarification(self):
        r = validate_voice_intent(intent("runner_database", {"query": "jamie"}, MIN_CONFIDENCE - 0.01), RUNNERS)
        self.assertEqual(r["status"], "clarify")

    def test_confidence_at_threshold_is_accepted(self):
        r = validate_voice_intent(intent("runner_database", {"query": "jamie"}, MIN_CONFIDENCE), RUNNERS)
        self.assertEqual(r["status"], "action")

    def test_missing_evidence_requires_clarification(self):
        r = validate_voice_intent(intent("runner_database", {"query": "jamie"}, evidence=[]), RUNNERS)
        self.assertEqual(r["status"], "clarify")

    def test_unsupported_intent_is_rejected(self):
        r = validate_voice_intent(intent("delete_runner", {"runner": "Jamie Lee"}), RUNNERS)
        self.assertEqual(r["status"], "clarify")

    def test_runner_selection_requires_one_exact_database_match(self):
        r = validate_voice_intent(intent("select_runner", {"runner": "Jamie Lee"}), RUNNERS)
        self.assertEqual(r["arguments"]["runner_id"], "R00044")
        self.assertEqual(validate_voice_intent(intent("select_runner", {"runner": "Alex Smith"}), RUNNERS)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("select_runner", {"runner": "Unknown Person"}), RUNNERS)["status"], "clarify")

    def test_familiarity_is_personal_not_database_creation(self):
        known = validate_voice_intent(intent("set_familiarity", {"value": "known"}), RUNNERS)
        unfamiliar = validate_voice_intent(intent("set_familiarity", {"value": "unfamiliar"}), RUNNERS)
        self.assertEqual(known["arguments"]["condition"], "known")
        self.assertEqual(unfamiliar["arguments"]["condition"], "unknown")
        self.assertNotIn("create_runner", unfamiliar["arguments"])

    def test_runner_search_preserves_query(self):
        r = validate_voice_intent(intent("runner_database", {"query": "Jamie Lee"}), RUNNERS)
        self.assertEqual(r["arguments"]["query"], "Jamie Lee")
        filtered = validate_voice_intent(intent("runner_database", {"query": "", "gender": "F", "min_races": 4}), RUNNERS)
        self.assertEqual(filtered["arguments"], {"query": "", "gender": "F", "min_races": 4})
        self.assertEqual(validate_voice_intent(intent("runner_database", {"gender": "X"}), RUNNERS)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("runner_database", {"min_races": 21}), RUNNERS)["status"], "clarify")

    def test_panel_action_must_match_the_spoken_open_or_close_verb(self):
        close_as_open = validate_voice_intent(
            intent("focus_panel", {"panel": "races"}, evidence=["close the races panel"]),
            RUNNERS, utterance="close the races panel",
        )
        open_as_close = validate_voice_intent(
            intent("close_panel", {"panel": "races"}, evidence=["open the races panel"]),
            RUNNERS, utterance="open the races panel",
        )
        opened = validate_voice_intent(
            intent("focus_panel", {"panel": "races"}, evidence=["show the races panel"]),
            RUNNERS, utterance="show the races panel",
        )
        closed = validate_voice_intent(
            intent("close_panel", {"panel": "races"}, evidence=["hide the races panel"]),
            RUNNERS, utterance="hide the races panel",
        )
        self.assertEqual(close_as_open["status"], "clarify")
        self.assertEqual(open_as_close["status"], "clarify")
        self.assertEqual(opened["status"], "action")
        self.assertEqual(closed["status"], "action")

    def test_panel_and_simulation_bounds_are_checked(self):
        self.assertEqual(validate_voice_intent(intent("focus_panel", {"panel": "races"}), RUNNERS)["status"], "action")
        self.assertEqual(validate_voice_intent(intent("focus_panel", {"panel": "secrets"}), RUNNERS)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("focus_panel", {"panel": "hologram"}), RUNNERS)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("focus_panel", {"panel": "record"}), RUNNERS)["status"], "action")
        close = validate_voice_intent(
            intent("close_panel", {"panel": "races"}, evidence=["close the races panel"]),
            RUNNERS, utterance="close the races panel",
        )
        self.assertEqual(close["status"], "action")
        self.assertEqual(validate_voice_intent(intent("simulate", {"temp_c": 41, "wind_kmh": 0}), RUNNERS)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("simulate", {"temp_c": 36, "wind_kmh": 0}), RUNNERS)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("simulate", {"temp_c": 22, "wind_kmh": 41}), RUNNERS)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("simulate", {"temp_c": 22, "wind_kmh": 15, "profile": "fast_start"}), RUNNERS)["status"], "action")

    def test_compound_panel_actions_are_clarified(self):
        utterance = "open the races panel and close the simulator"
        result = validate_voice_intent(
            intent("focus_panel", {"panel": "races"}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(result["status"], "clarify")
        self.assertIn("more than one action", result["message"])

    def test_compound_runner_selection_and_familiarity_are_clarified(self):
        utterance = "open Jamie Lee and mark this runner unfamiliar to me"
        result = validate_voice_intent(
            intent("select_runner", {"runner": "Jamie Lee"}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(result["status"], "clarify")

    def test_compound_database_search_and_runner_selection_are_clarified(self):
        utterance = "open the runner database and select Jamie Lee"
        result = validate_voice_intent(
            intent("runner_database", {"query": "Jamie Lee"}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(result["status"], "clarify")

    def test_compound_action_and_question_are_clarified(self):
        utterance = "open the races panel and explain why the range is wide"
        result = validate_voice_intent(
            intent("focus_panel", {"panel": "races"}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(result["status"], "clarify")

    def test_conflicting_simulation_temperatures_are_clarified(self):
        utterance = "simulate at 25 degrees and then at 30 degrees"
        result = validate_voice_intent(
            intent("simulate", {"temp_c": 25, "wind_kmh": 0, "profile": "on_plan"},
                   evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(result["status"], "clarify")

    def test_simulation_may_open_its_own_panel(self):
        utterance = "simulate at 25 degrees and open the simulator"
        result = validate_voice_intent(
            intent("simulate", {"temp_c": 25, "wind_kmh": 0, "profile": "on_plan"},
                   evidence=[utterance]),
            RUNNERS, utterance=utterance,
            ui_state={"temp_c": 12, "wind_kmh": 0, "profile": "on_plan"},
        )
        self.assertEqual(result["status"], "action")

    def test_state_dependent_actions_are_checked(self):
        self.assertEqual(validate_voice_intent(intent("undo"), RUNNERS, can_undo=False)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("undo"), RUNNERS, can_undo=True)["status"], "action")
        self.assertEqual(validate_voice_intent(intent("reveal"), RUNNERS, can_reveal=False)["status"], "clarify")

    def test_evidence_and_search_targets_must_be_grounded_in_transcript(self):
        utterance = "find Jamie Lee in the runner database"
        invented_query = validate_voice_intent(
            intent("runner_database", {"query": "Alex Smith"}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(invented_query["status"], "clarify")
        invented_evidence = validate_voice_intent(
            intent("runner_database", {"query": "Jamie Lee"}, evidence=["open the database"]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(invented_evidence["status"], "clarify")

    def test_runner_search_cannot_drop_a_spoken_name_or_id(self):
        utterance = "find Jamie Lee in the runner database"
        result = validate_voice_intent(
            intent("runner_database", {"query": ""}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(result["status"], "clarify")
        self.assertIn("wasn't included in the search", result["message"])

    def test_runner_search_cannot_drop_a_spoken_gender_filter(self):
        utterance = "show female runners"
        result = validate_voice_intent(
            intent("runner_database", {"gender": ""}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(result["status"], "clarify")
        self.assertIn("female-runner filter", result["message"])

    def test_runner_search_cannot_drop_a_spoken_minimum_race_filter(self):
        utterance = "show runners with at least six marathons"
        result = validate_voice_intent(
            intent("runner_database", {"min_races": 0}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(result["status"], "clarify")
        self.assertIn("minimum of 6", result["message"])

    def test_runner_database_does_not_use_familiarity_as_a_filter(self):
        utterance = "find unfamiliar runners"
        result = validate_voice_intent(
            intent("runner_database", {}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(result["status"], "clarify")
        self.assertIn("not a runner-database filter", result["message"])

    def test_model_cannot_invent_gender_or_minimum_race_filters(self):
        utterance = "find runners in the database"
        female = validate_voice_intent(
            intent("runner_database", {"gender": "F"}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        races = validate_voice_intent(
            intent("runner_database", {"min_races": 4}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(female["status"], "clarify")
        self.assertEqual(races["status"], "clarify")

    def test_zero_asr_confidence_is_not_treated_as_missing(self):
        utterance = "find Jamie Lee in the runner database"
        result = validate_voice_intent(
            intent("runner_database", {"query": "Jamie Lee"}, evidence=[utterance]),
            RUNNERS, utterance=utterance, transcript_confidence=0.0,
        )
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(result["confidence"], 0.0)

    def test_asr_confidence_is_combined_with_model_confidence(self):
        utterance = "find Jamie Lee in the runner database"
        result = validate_voice_intent(
            intent("runner_database", {"query": "Jamie Lee"}, evidence=[utterance]),
            RUNNERS, utterance=utterance, transcript_confidence=0.42,
        )
        self.assertEqual(result["status"], "clarify")
        self.assertLess(result["confidence"], MIN_CONFIDENCE)

    def test_rule_fallback_requires_explicit_personal_familiarity_language(self):
        self.assertEqual(ui_commands("unknown runner")[0], [])
        self.assertEqual(ui_commands("find an unknown runner")[0], [])
        self.assertEqual(ui_commands("I don't know this runner")[0], [{"type": "condition", "value": "unknown"}])
        self.assertEqual(ui_commands("mark this runner unfamiliar to me")[0], [{"type": "condition", "value": "unknown"}])
        self.assertEqual(ui_commands("I am familiar with this runner")[0], [{"type": "condition", "value": "known"}])

    def test_familiarity_filter_words_are_not_active_runner_changes(self):
        unfamiliar = validate_voice_intent(
            intent("set_familiarity", {"value": "unfamiliar"}, evidence=["find unfamiliar runners"]),
            RUNNERS, utterance="find unfamiliar runners",
        )
        known = validate_voice_intent(
            intent("set_familiarity", {"value": "known"}, evidence=["show familiar runners"]),
            RUNNERS, utterance="show familiar runners",
        )
        self.assertEqual(unfamiliar["status"], "clarify")
        self.assertEqual(known["status"], "clarify")

    def test_unknown_runner_search_is_not_misread_as_familiarity(self):
        utterance = "find an unknown runner"
        result = validate_voice_intent(
            intent("set_familiarity", {"value": "unknown"}, evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(result["status"], "clarify")

    def test_unknown_and_known_wording_maps_to_familiarity_values(self):
        unknown = validate_voice_intent(
            intent("set_familiarity", {"value": "unknown"}, evidence=["mark this runner as unknown"]),
            RUNNERS, utterance="mark this runner as unknown",
        )
        known = validate_voice_intent(
            intent("set_familiarity", {"value": "known"}, evidence=["mark this runner as known"]),
            RUNNERS, utterance="mark this runner as known",
        )
        self.assertEqual(unknown["arguments"]["condition"], "unknown")
        self.assertEqual(known["arguments"]["condition"], "known")

    def test_demo_runner_alias_is_resolved_to_the_active_demo_session(self):
        rows = RUNNERS + [{"runner_id": "demo", "name": "Demo runner"}]
        utterance = "mark Demo runner unfamiliar to me"
        result = validate_voice_intent(
            intent("set_familiarity", {"value": "unfamiliar", "runner": "Demo runner"},
                   evidence=[utterance]),
            rows, utterance=utterance, current_runner_id="R00042",
            current_runner_ids=["R00042", "R00044", "demo"],
        )
        self.assertEqual(result["status"], "action")
        self.assertEqual(result["arguments"]["condition"], "unknown")

    def test_familiarity_change_cannot_target_a_different_active_runner(self):
        result = validate_voice_intent(
            intent("set_familiarity", {"value": "unfamiliar", "runner": "Jamie Lee"},
                   evidence=["mark Jamie Lee as unfamiliar to me"]),
            RUNNERS, utterance="mark Jamie Lee as unfamiliar to me", current_runner_id="R00042",
        )
        self.assertEqual(result["status"], "clarify")
        self.assertIn("open Jamie Lee first", result["message"])

    def test_simulation_keeps_unspecified_current_ui_values(self):
        ui_state = {"temp_c": 18, "wind_kmh": 12, "profile": "late_fade"}
        utterance = "simulate at 25 degrees"
        valid = validate_voice_intent(
            intent("simulate", {"temp_c": 25, "wind_kmh": 12, "profile": "late_fade"},
                   evidence=[utterance]),
            RUNNERS, utterance=utterance, ui_state=ui_state,
        )
        invented = validate_voice_intent(
            intent("simulate", {"temp_c": 25, "wind_kmh": 0, "profile": "on_plan"},
                   evidence=[utterance]),
            RUNNERS, utterance=utterance, ui_state=ui_state,
        )
        self.assertEqual(valid["status"], "action")
        self.assertEqual(invented["status"], "clarify")

    def test_weather_adjective_does_not_invent_a_temperature(self):
        utterance = "simulate a hot day"
        result = validate_voice_intent(
            intent("simulate", {"temp_c": 27, "wind_kmh": 0, "profile": "on_plan"},
                   evidence=[utterance]),
            RUNNERS, utterance=utterance,
        )
        self.assertEqual(result["status"], "clarify")

    def test_voice_schema_carries_current_ui_state(self):
        request = Command.model_validate({
            "text": "simulate at 25 degrees", "source": "voice",
            "voice": {"confidence": 0.9, "ui_state": {
                "temp_c": 18, "wind_kmh": 12, "profile": "late_fade",
            }},
        })
        self.assertEqual(request.voice.ui_state.temp_c, 18)
        self.assertEqual(request.voice.ui_state.profile, "late_fade")


class FakeIntentClient:
    def __init__(self, payload):
        self.payload = payload

    def chat(self, model, messages, **kwargs):
        return SimpleNamespace(text=json.dumps(self.payload))


class FakeChatModel:
    label = "fake Boson chat"

    def __init__(self, answer):
        self.answer = answer
        self.messages = []

    def complete(self, messages, tools):
        self.messages = messages
        return ModelReply(self.answer)


class FakeAI:
    online = True

    def __init__(self, payload, chat_model=None):
        self.client = FakeIntentClient(payload)
        self._chat_model = chat_model

    def config(self, task):
        return "higgs-realtime", 0.0

    def chat_model(self):
        return self._chat_model or FakeChatModel("I understood the request.")


class VoiceIntentExecutionTests(unittest.TestCase):
    def service(self, payload):
        coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        service = StudioService(coach, settings=Settings())
        sid, _ = service.create(condition="known")
        service.ai_factory = lambda: FakeAI(payload)
        return service, sid, coach

    def test_voice_freeform_reply_keeps_pending_context_and_is_visible(self):
        coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        chat = FakeChatModel("A trial run is not one of her marathon records. A leg injury could affect performance "
                             "depending on its severity and recovery, but I won't change marathon weights from it.")
        ai = FakeAI(intent("none"), chat)
        service = StudioService(coach, settings=Settings(), ai_factory=lambda: ai)
        sid, _ = service.create(condition="unknown")
        live = service.sessions.entry(sid).value
        live.engine.pending = {"loop": "clarify", "message": "Which marathon was the injury?",
                               "data": {"choices": [{"label": "None of them"}]}}
        utterance = "I meant the 2026 trial run; she had a leg injury"
        result = service.command(sid, utterance, source="voice")
        self.assertIn("A trial run is not one of her marathon records", result["reply"])
        self.assertIn("Which marathon was the injury?", chat.messages[0]["content"])
        self.assertIn("Existing marathon records:", chat.messages[0]["content"])
        self.assertEqual(live.hud.history[-2]["content"], utterance)
        self.assertIn("leg injury", live.hud.history[-1]["content"])

    def test_intent_classifier_runs_for_paraphrases_but_not_for_race_comments(self):
        service, sid, _ = self.service(intent("none"))
        live = service.sessions.entry(sid).value
        seen = []
        original = service._interpret_voice_intent

        def spy(text, live_, signals):
            seen.append(text)
            return original(text, live_, signals)

        service._interpret_voice_intent = spy
        race = live.hud.steer.races[0]
        service.command(sid, f"{race['event']} {race['year']} should count for less", source="voice")
        self.assertEqual(seen, [])                       # one steering call, no second classifier call
        service.command(sid, "could you look up Jamie Lee for me", source="voice")
        self.assertEqual(seen, ["could you look up Jamie Lee for me"])
        service.command(sid, "could you look up Jamie Lee for me", source="typed")
        self.assertEqual(len(seen), 1)                   # typed text never uses the voice classifier

    def test_voice_action_confirmation_is_deferred_until_execution(self):
        service, sid, _ = self.service(intent(
            "focus_panel", {"panel": "races"}, evidence=["show the races panel"],
        ))
        result = service.command(sid, "show the races panel", source="voice")
        self.assertEqual(result["voice_intent"]["status"], "action")
        self.assertEqual(result["voice_intent"]["arguments"]["panel"], "races")
        self.assertTrue(result["voice_action_pending"])
        self.assertEqual(result["adapt"]["speak"], "")

    def test_low_transcript_confidence_does_not_execute_a_voice_action(self):
        service, sid, _ = self.service(intent(
            "focus_panel", {"panel": "races"}, evidence=["show the races panel"],
        ))
        result = service.command(
            sid, "show the races panel", source="voice", voice={"confidence": 0.4},
        )
        self.assertEqual(result["voice_intent"]["status"], "clarify")
        self.assertEqual(result["actions"], [])

    def test_compound_voice_ui_actions_are_clarified_before_execution(self):
        service, sid, _ = self.service(intent("none"))
        result = service.command(
            sid, "open the races panel and close the simulator", source="voice",
        )
        self.assertEqual(result["actions"], [])
        self.assertFalse(result["voice_action_pending"])
        self.assertIn("more than one request", result["reply"])

    def test_voice_steering_change_returns_a_confirmable_refresh_action(self):
        service, sid, _ = self.service(intent("none"))
        result = service.command(sid, "trust the recent races more", source="voice")
        self.assertTrue(result["voice_action_pending"])
        self.assertIn({"type": "refresh"}, result["actions"])
        self.assertTrue(result["turn"]["applied"])

    def test_rule_fallback_voice_action_still_defers_confirmation(self):
        service, sid, _ = self.service(intent("none"))
        result = service.command(sid, "show the races panel", source="voice")
        self.assertTrue(result["voice_action_pending"])
        self.assertEqual(result["adapt"]["speak"], "")

    def test_voice_runner_search_preserves_explicit_gender_and_custom_minimum(self):
        utterance = "show female runners with at least six marathons"
        service, sid, _ = self.service(intent(
            "runner_database", {"query": "", "gender": "F", "min_races": 6},
            evidence=[utterance],
        ))
        result = service.command(sid, utterance, source="voice")
        self.assertIn(
            {"type": "runner_database", "query": "", "gender": "F", "min_races": 6},
            result["actions"],
        )

    def test_voice_runner_search_executes_database_navigation_with_query(self):
        service, sid, _ = self.service(intent(
            "runner_database", {"query": "Jamie Lee"},
            evidence=["find Jamie Lee in the runner database"],
        ))
        result = service.command(sid, "find Jamie Lee in the runner database", source="voice")
        self.assertIn({"type": "runner_database", "query": "Jamie Lee", "gender": "", "min_races": 0}, result["actions"])
        self.assertEqual(result["voice_intent"]["confidence"], 0.95)
        self.assertTrue(result["voice_intent"]["evidence"])

    def test_voice_familiarity_executes_existing_runner_condition(self):
        service, sid, _ = self.service(intent(
            "set_familiarity", {"value": "unfamiliar"},
            evidence=["I don't know this runner"],
        ))
        result = service.command(sid, "I don't know this runner", source="voice")
        self.assertEqual(result["adapt"]["ui"]["condition"], "unknown")
        self.assertTrue(result["voice_action_pending"])
        self.assertEqual(result["actions"], [{"type": "condition", "value": "unknown"}])

    def test_voice_close_panel_is_mapped_to_a_close_action(self):
        utterance = "hide the races panel"
        service, sid, _ = self.service(intent(
            "close_panel", {"panel": "races"}, evidence=[utterance],
        ))
        result = service.command(sid, utterance, source="voice")
        self.assertEqual(result["actions"], [{"type": "close", "panel": "races"}])

    def test_boson_unavailable_simulation_preserves_unspoken_controls(self):
        service, sid, _ = self.service(intent("none"))
        service.ai_factory = lambda: SimpleNamespace(online=False)
        ui_state = {"temp_c": 18, "wind_kmh": 12, "profile": "late_fade"}
        result = service.command(
            sid, "simulate at 25 degrees", source="voice",
            voice={"confidence": 0.95, "ui_state": ui_state},
        )
        self.assertEqual(result["actions"], [{
            "type": "simulate", "temp_c": 25.0, "wind_kmh": 12.0, "profile": "late_fade",
        }])
        self.assertEqual(result["simulation"]["summary"]["temp_c"], 25.0)

    def test_offline_question_about_simulation_does_not_run_it(self):
        service, sid, _ = self.service(intent("none"))
        service.ai_factory = lambda: SimpleNamespace(online=False)
        result = service.command(
            sid, "what does the simulation do?", source="voice",
            voice={"confidence": 0.95, "ui_state": {
                "temp_c": 18, "wind_kmh": 12, "profile": "late_fade",
            }},
        )
        self.assertEqual(result["actions"], [])
        self.assertIsNone(result["simulation"])
        self.assertIn("Do you want me to run", result["reply"])

    def test_boson_unavailable_does_not_guess_temperature_from_hot(self):
        service, sid, _ = self.service(intent("none"))
        service.ai_factory = lambda: SimpleNamespace(online=False)
        result = service.command(
            sid, "simulate a hot day", source="voice",
            voice={"confidence": 0.95, "ui_state": {
                "temp_c": 18, "wind_kmh": 12, "profile": "late_fade",
            }},
        )
        self.assertEqual(result["actions"], [])
        self.assertIsNone(result["simulation"])
        self.assertIn("specific temperature", result["reply"])

    def test_offline_fallback_rejects_unsupported_legacy_view_actions(self):
        service, sid, _ = self.service(intent("none"))
        service.ai_factory = lambda: SimpleNamespace(online=False)
        result = service.command(sid, "switch to runner view", source="voice")
        self.assertEqual(result["actions"], [])
        self.assertIn("isn't available in PaceSense Studio", result["reply"])

    def test_voice_track_record_navigation_targets_the_profile_section(self):
        utterance = "open the track record"
        service, sid, _ = self.service(intent(
            "focus_panel", {"panel": "record"}, evidence=[utterance],
        ))
        result = service.command(sid, utterance, source="voice")
        self.assertEqual(result["actions"], [{"type": "focus", "panel": "record"}])

    def test_voice_panel_navigation_uses_a_real_studio_panel(self):
        service, sid, _ = self.service(intent(
            "focus_panel", {"panel": "races"}, evidence=["show the races panel"],
        ))
        result = service.command(sid, "show the races panel", source="voice")
        self.assertEqual(result["actions"], [{"type": "focus", "panel": "races"}])

    def test_voice_simulation_executes_with_validated_parameters(self):
        service, sid, _ = self.service(intent(
            "simulate", {"temp_c": 25, "wind_kmh": 10, "profile": "on_plan"},
            evidence=["simulate at 25 degrees and 10 km/h headwind"],
        ))
        result = service.command(sid, "simulate at 25 degrees and 10 km/h headwind", source="voice")
        self.assertEqual(result["simulation"]["summary"]["temp_c"], 25.0)
        self.assertEqual(result["actions"][0]["wind_kmh"], 10.0)

    def test_voice_runner_selection_uses_verified_database_id(self):
        service, sid, coach = self.service(intent("none"))
        live = service.sessions.entry(sid).value
        runner = next(r for r in coach.runners if r.runner_id != live.hud.runner_id)
        target = runner.runner_id
        service.ai_factory = lambda: FakeAI(intent(
            "select_runner", {"runner": target}, evidence=[f"open runner {target}"],
        ))
        result = service.command(sid, f"open runner {target}", source="voice")
        name = CoachService._summary(runner, False).get("name") or runner.runner_id
        self.assertEqual(result["actions"], [
            {"type": "runner", "id": runner.runner_id, "name": name, "condition": "ask"},
        ])

    def test_selecting_the_current_runner_keeps_its_existing_familiarity(self):
        service, sid, _ = self.service(intent("none"))
        live = service.sessions.entry(sid).value
        target = live.hud.runner_id
        service.ai_factory = lambda: FakeAI(intent(
            "select_runner", {"runner": target}, evidence=[f"open runner {target}"],
        ))
        result = service.command(sid, f"open runner {target}", source="voice")
        self.assertEqual(result["actions"][0]["condition"], "known")

    def test_compound_voice_action_is_clarified_even_when_boson_proposes_one(self):
        utterance = "open the races panel and close the simulator"
        service, sid, _ = self.service(intent(
            "focus_panel", {"panel": "races"}, evidence=[utterance],
        ))
        result = service.command(sid, utterance, source="voice")
        self.assertEqual(result["actions"], [])
        self.assertEqual(result["voice_intent"]["status"], "clarify")
        self.assertIn("more than one action", result["reply"])

    def test_unmatched_runner_selection_does_not_execute(self):
        service, sid, _ = self.service(intent(
            "select_runner", {"runner": "Unknown Person"}, evidence=["open Unknown Person"],
        ))
        result = service.command(sid, "open Unknown Person", source="voice")
        self.assertEqual(result["actions"], [])
        self.assertIn("couldn't find an exact existing runner match", result["reply"])

    def test_runner_directory_filters_are_executed_by_the_api(self):
        coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        client = Client(create_app(StudioService(coach, settings=Settings())))
        rows = client.get("/api/v1/runners?gender=F&min_races=4").json()
        self.assertTrue(all(row["gender"] == "F" and row["n_races"] >= 4 for row in rows))
        self.assertEqual(client.get("/api/v1/runners?gender=X").status, 422)
        self.assertEqual(client.get("/api/v1/runners?min_races=21").status, 422)




if __name__ == "__main__":
    unittest.main()
