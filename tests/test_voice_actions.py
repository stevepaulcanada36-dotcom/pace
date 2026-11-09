"""Validation tests for AI-interpreted voice actions."""
import json
import unittest
from types import SimpleNamespace

from pacesense.coach import CoachService
from pacesense.storage import Repository
from pacesense.studio import Settings, StudioService
from pacesense.studio.app import create_app
from tests.asgi import Client
from tests import helpers

from pacesense.hud.voice_actions import MIN_CONFIDENCE, validate_voice_intent


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

    def test_panel_and_simulation_bounds_are_checked(self):
        self.assertEqual(validate_voice_intent(intent("focus_panel", {"panel": "races"}), RUNNERS)["status"], "action")
        self.assertEqual(validate_voice_intent(intent("focus_panel", {"panel": "secrets"}), RUNNERS)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("focus_panel", {"panel": "hologram"}), RUNNERS)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("simulate", {"temp_c": 41, "wind_kmh": 0}), RUNNERS)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("simulate", {"temp_c": 22, "wind_kmh": 15, "profile": "fast_start"}), RUNNERS)["status"], "action")

    def test_state_dependent_actions_are_checked(self):
        self.assertEqual(validate_voice_intent(intent("undo"), RUNNERS, can_undo=False)["status"], "clarify")
        self.assertEqual(validate_voice_intent(intent("undo"), RUNNERS, can_undo=True)["status"], "action")
        self.assertEqual(validate_voice_intent(intent("reveal"), RUNNERS, can_reveal=False)["status"], "clarify")


class FakeIntentClient:
    def __init__(self, payload):
        self.payload = payload

    def chat(self, model, messages, **kwargs):
        return SimpleNamespace(text=json.dumps(self.payload))


class FakeAI:
    online = True

    def __init__(self, payload):
        self.client = FakeIntentClient(payload)

    def config(self, task):
        return "higgs-realtime", 0.0


class VoiceIntentExecutionTests(unittest.TestCase):
    def service(self, payload):
        coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        service = StudioService(coach, settings=Settings())
        sid, _ = service.create(condition="known")
        service.ai_factory = lambda: FakeAI(payload)
        return service, sid, coach

    def test_voice_runner_search_executes_database_navigation_with_query(self):
        service, sid, _ = self.service(intent("runner_database", {"query": "Jamie Lee"}))
        result = service.command(sid, "find Jamie Lee in the runner database", source="voice")
        self.assertIn({"type": "runner_database", "query": "Jamie Lee", "gender": "", "min_races": 0}, result["actions"])
        self.assertEqual(result["voice_intent"]["confidence"], 0.95)
        self.assertTrue(result["voice_intent"]["evidence"])

    def test_voice_familiarity_executes_existing_runner_condition(self):
        service, sid, _ = self.service(intent("set_familiarity", {"value": "unfamiliar"}))
        result = service.command(sid, "I don't know this runner", source="voice")
        self.assertEqual(result["adapt"]["ui"]["condition"], "unknown")
        self.assertEqual(result["actions"], [{"type": "condition", "value": "unknown"}])

    def test_voice_runner_selection_uses_verified_database_id(self):
        service, sid, coach = self.service(intent("none"))
        runner = coach.runners[0]
        target = runner.runner_id
        service.ai_factory = lambda: FakeAI(intent("select_runner", {"runner": target}))
        result = service.command(sid, f"open runner {target}", source="voice")
        name = CoachService._summary(runner, False).get("name") or runner.runner_id
        self.assertEqual(result["actions"], [{"type": "runner", "id": runner.runner_id, "name": name}])

    def test_ambiguous_runner_selection_does_not_execute(self):
        service, sid, _ = self.service(intent("select_runner", {"runner": "Alex Smith"}))
        result = service.command(sid, "open Alex Smith", source="voice")
        self.assertEqual(result["actions"], [])
        self.assertIn("more than one", result["reply"])

    def test_runner_directory_filters_are_executed_by_the_api(self):
        coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        client = Client(create_app(StudioService(coach, settings=Settings())))
        rows = client.get("/api/v1/runners?gender=F&min_races=4").json()
        self.assertTrue(all(row["gender"] == "F" and row["n_races"] >= 4 for row in rows))
        self.assertEqual(client.get("/api/v1/runners?gender=X").status, 422)
        self.assertEqual(client.get("/api/v1/runners?min_races=21").status, 422)




if __name__ == "__main__":
    unittest.main()
