"""Boson integration, per-task settings, voice pipeline and evaluation lab tests."""
import json
import unittest

from pacesense.coach import CoachService
from pacesense.domain import ValidationError
from pacesense.llm import evals
from pacesense.llm.forms import FORMS, facts
from pacesense.llm.openrouter import LLMError, OpenRouterClient, parse_json_answer
from pacesense.llm.service import AIService, AIUnavailable
from pacesense.llm.tasks import TASKS, ModelSettings
from pacesense.storage import Repository

from .fake_openrouter import FakeOpenRouter, chat_reply, smart_handler
from .helpers import model, runners


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOpenRouter()
        self.c = self.fake.client()

    def test_needs_key(self):
        with self.assertRaises(LLMError):
            OpenRouterClient("")

    def test_chat_request_and_result(self):
        self.fake.chats = [chat_reply("hi there", cost=0.00042, model="openai/gpt-4o-mini-2024")]
        r = self.c.chat("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}], temperature=0.3, json_mode=True)
        self.assertEqual(r.text, "hi there")
        self.assertEqual((r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cost), (100, 20, 0.00042))
        self.assertEqual(r.model, "openai/gpt-4o-mini-2024")
        req = self.fake.requests[0]
        self.assertEqual(req["headers"]["Authorization"], "Bearer sk-or-test")
        self.assertEqual(req["headers"]["X-Title"], "PaceSense")
        self.assertEqual(req["body"]["usage"], {"include": True})
        self.assertEqual(req["body"]["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", req["body"])

    def test_tool_calls_parsed(self):
        self.fake.chats = [chat_reply(None, [("predict", {"runner_id": "R1"})])]
        r = self.c.chat("m/x", [], tools=[{"type": "function"}])
        self.assertEqual(r.tool_calls, [{"id": "call_0", "name": "predict", "arguments": {"runner_id": "R1"}}])

    def test_errors(self):
        self.fake.chats = [(402, "Insufficient credits"), (200, "ignored")]
        with self.assertRaises(LLMError) as e:
            self.c.chat("m/x", [])
        self.assertEqual(e.exception.status, 402)
        self.assertIn("Insufficient credits", str(e.exception))

    def test_transcribe_rejects_empty_and_oversized_audio_before_network(self):
        with self.assertRaisesRegex(LLMError, "recording is empty"):
            self.c.transcribe("openai/whisper-1", b"")
        with self.assertRaisesRegex(LLMError, "larger than 25 MB"):
            self.c.transcribe("openai/whisper-1", b"x" * (25 * 1024 * 1024 + 1))
        self.assertEqual(self.fake.requests, [])

    def test_transcribe_endpoint(self):
        self.fake.transcripts = ["three hours twenty"]
        t = self.c.transcribe("openai/whisper-1", b"RIFFdata", "wav", "en")
        self.assertEqual((t.text, t.route, t.usage.audio_seconds, t.usage.cost), ("three hours twenty", "transcriptions", 3.0, 0.0003))
        body = self.fake.bodies("/audio/transcriptions")[0]
        self.assertEqual(body["input_audio"]["format"], "wav")
        self.assertEqual(body["language"], "en")

    def test_transcribe_falls_back_to_chat_audio(self):
        self.fake.transcripts = [(400, "model does not support transcription")]
        self.fake.chats = [chat_reply("hello from gemini")]
        t = self.c.transcribe("google/gemini-2.5-flash", b"abc", "wav")
        self.assertEqual((t.text, t.route), ("hello from gemini", "chat"))
        parts = self.fake.bodies("/chat/completions")[0]["messages"][0]["content"]
        self.assertEqual(parts[1]["type"], "input_audio")
        self.assertEqual(parts[1]["input_audio"]["data"], "YWJj")       # base64 of b"abc"

    def test_transcribe_auth_error_does_not_fall_back(self):
        self.fake.transcripts = [(401, "bad key")]
        with self.assertRaises(LLMError):
            self.c.transcribe("openai/whisper-1", b"abc")
        self.assertEqual(self.fake.bodies("/chat/completions"), [])

    def test_models(self):
        ms = self.c.models()
        self.assertEqual([m["id"] for m in ms], ["openai/gpt-4o-mini", "openai/whisper-1"])
        self.assertEqual(ms[0]["prompt_usd_per_mtok"], 0.15)
        self.assertTrue(ms[0]["tools"])
        self.assertEqual(ms[1]["inputs"], ["audio"])
        self.assertEqual([r["path"] for r in self.fake.requests],
                         ["/models", "/models?output_modalities=transcription"])

    def test_parse_json_answer(self):
        self.assertEqual(parse_json_answer('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_answer('Sure! {"a": 2} hope it helps'), {"a": 2})
        for bad in ("no json", "[1, 2]", "{broken"):
            with self.assertRaises(LLMError):
                parse_json_answer(bad)


class SettingsTests(unittest.TestCase):
    def test_priority_settings_env_default(self):
        repo = Repository(":memory:")
        s = ModelSettings(repo, env={"PACESENSE_MODEL_REFINE": "higgs-realtime",
                                     "PACESENSE_TEMPERATURE_REFINE": "0.7"})
        self.assertEqual(s.get("extract"), {"model": TASKS["extract"].default_model, "temperature": 0.0,
                                            "source": "default"})
        self.assertEqual(s.get("refine"), {"model": "higgs-realtime", "temperature": 0.7, "source": "default"})
        s.update("refine", model="higgs-realtime", temperature=0.2)
        self.assertEqual(s.get("refine"), {"model": "higgs-realtime", "temperature": 0.2, "source": "settings"})
        self.assertEqual(ModelSettings(repo, env={}).get("refine")["model"], "higgs-realtime")   # saved in the db
        s.reset("refine")
        self.assertEqual(s.get("refine")["source"], "default")
        for bad in ({"model": "gpt4"}, {"model": "a/b c"}, {"temperature": 3}):
            with self.assertRaises(ValidationError):
                s.update("refine", **bad)
        with self.assertRaises(ValidationError):
            s.update("nope", model="a/b")

    def test_every_task_uses_the_boson_model(self):
        repo = Repository(":memory:")
        fake = FakeOpenRouter()
        fake.chat_handler = smart_handler
        repo.set_setting("transcription_backend", "boson")
        ai = AIService(repo, env={}, client=fake.client())
        for task in TASKS:
            ai.settings.update(task, model="higgs-realtime")
        ai.voice("runner", b"audio", "wav")
        ai.chat_model().complete([{"role": "user", "content": "hi"}], [])
        used = [r["body"]["model"] for r in fake.requests]
        self.assertEqual(used, ["higgs-realtime", "higgs-realtime", "higgs-realtime", "higgs-realtime"])


class ServiceTests(unittest.TestCase):
    def test_local_transcription_can_be_selected_as_fallback(self):
        repo = Repository(":memory:")
        ai = AIService(repo, env={})
        repo.set_setting("transcription_backend", "local")
        self.assertEqual(ai.transcription_backend, "local")
        from pacesense.llm.openrouter import Transcript
        with __import__("unittest.mock", fromlist=["patch"]).patch(
                "pacesense.llm.local_whisper.transcribe",
                return_value=Transcript("hello", model="faster-whisper/base", route="local")) as local:
            result = ai.transcribe(b"audio", "wav")
        self.assertEqual(result.text, "hello")
        local.assert_called_once_with(b"audio", "wav", None)
        self.assertIsNone(ai.client)

    def setUp(self):
        self.repo = Repository(":memory:")
        self.fake = FakeOpenRouter()
        self.fake.chat_handler = smart_handler
        self.repo.set_setting("transcription_backend", "boson")
        self.ai = AIService(self.repo, env={}, client=self.fake.client())

    def test_offline_without_key(self):
        self.repo.set_setting("transcription_backend", "boson")
        ai = AIService(self.repo, env={})
        self.assertFalse(ai.online)
        self.assertIn("No BOSON_API_KEY", ai.why_offline())
        with self.assertRaises(AIUnavailable):
            ai.transcribe(b"x")
        ex = ai.extract("runner", "Add runner Sam Lee, f, 41: Toronto 2022 4:01:10")
        self.assertEqual(ex.fields["name"], "Sam Lee")
        r = ai.refine("question", "um is my plan ok")
        self.assertEqual(r.text, "Is my plan ok.")
        self.assertIn("Offline", r.changes[0])
        self.assertEqual(type(ai.chat_model()).__name__, "OfflineModel")
        self.assertFalse(AIService(self.repo, api_key="k", env={"PACESENSE_AI": "offline"}).online)

    def test_voice_pipeline(self):
        self.fake.transcripts = ["uh new runner anna de vries female 34 amsterdam 2023 3:35:48 rotterdam 2024 3h33m11"]
        v = self.ai.voice("runner", b"audio", "wav")
        self.assertIn("anna de vries", v.transcript.text)
        f = v.extraction.fields
        self.assertEqual((f["name"], f["gender"], f["age"]), ("Anna de Vries", "F", 34))
        self.assertEqual(f["races"][1]["finish"], "3:33:11")                         # corrected by the check
        self.assertTrue(any("3h33m11" in c for c in v.extraction.check.corrections))
        self.assertTrue(v.refinement.text)
        self.assertEqual(v.extraction.step.usage.cost, 0.0001)

    def test_refine_flags_changed_numbers(self):
        self.fake.chats = [chat_reply(json.dumps({"refined": "Finished in 3:25:04 on a hot day.", "changes": [],
                                                  "questions": ["Which race?"]}))]
        r = self.ai.refine("result", "finished 3:21:04 hot day walked after 35k")
        self.assertEqual(r.added, ["3:25:04"])
        self.assertEqual(set(r.dropped), {"3:21:04", "35"})
        self.assertFalse(r.keeps_facts)
        self.assertEqual(r.questions, ["Which race?"])

    def test_extract_bad_json_is_an_error(self):
        self.fake.chats = [chat_reply("I think the finish was 3:21")]
        with self.assertRaises(LLMError):
            self.ai.extract("result", "3:21")

    def test_forms_check_fields(self):
        self.assertEqual(FORMS["result"].check({"finish": "3h21m4s"}).fields["finish"], "3:21:04")
        self.assertTrue(FORMS["result"].check({"finish": "9:00:00"}).problems)
        p = FORMS["plan_details"].check({"target_race": "  Toronto  ", "race_date": "18 Oct", "notes": ""})
        self.assertEqual(p.fields["target_race"], "Toronto")
        self.assertTrue(p.problems)
        self.assertEqual(facts("3h35 and 3:35:00, 32 km"), {"t12900", "n32"})


class ReviewRegressionTests(unittest.TestCase):
    """Problems found in review: each must stay fixed."""

    def setUp(self):
        self.repo = Repository(":memory:")
        self.repo.set_setting("transcription_backend", "boson")
        self.fake = FakeOpenRouter()
        self.ai = AIService(self.repo, env={}, client=self.fake.client())

    def test_badly_shaped_runner_json_is_a_problem_not_a_crash(self):
        self.fake.chats = [chat_reply(json.dumps({"name": "X", "gender": "f", "races": ["Amsterdam 2023 3:35"]})),
                           chat_reply(json.dumps({"name": "X", "gender": "f", "races": [
                               {"event": "A", "year": 2020, "finish": "3:30:00", "splits": 7}]})),
                           chat_reply(json.dumps({"name": "X", "races": "none"}))]
        for _ in range(3):
            ex = self.ai.extract("runner", "anything")
            self.assertTrue(ex.check.problems)

    def test_failed_refine_keeps_transcript_and_fields(self):
        self.fake.transcripts = ["finished in 3:21:04"]
        self.fake.chats = [chat_reply(json.dumps({"finish": "3:21:04"})), (502, "provider error")]
        v = self.ai.voice("result", b"x")
        self.assertEqual(v.transcript.text, "finished in 3:21:04")
        self.assertEqual(v.extraction.fields["finish"], "3:21:04")
        self.assertIsNone(v.refinement)
        self.assertIn("provider error", v.errors[0])

    def test_rewritten_dates_and_times_are_not_flagged(self):
        self.fake.chats = [chat_reply(json.dumps({"refined": "Toronto Waterfront Marathon on 2026-10-18; Berlin "
                                                  "2019 in 3:58:00.", "changes": [], "questions": []}))]
        r = self.ai.refine("plan_details", "toronto waterfront marathon on october 18th 2026, berlin 2019 in 3 hours 58")
        self.assertTrue(r.keeps_facts, (r.added, r.dropped))

    def test_temperature_change_does_not_pin_the_model(self):
        s = ModelSettings(self.repo, env={"PACESENSE_MODEL_REFINE": "google/gemini-2.5-flash"})
        s.update("refine", temperature=0.9)
        self.assertEqual(s.get("refine")["source"], "default")
        self.assertTrue(s.has_saved("refine"))
        s.reset("refine")
        self.assertFalse(s.has_saved("refine"))

    def test_lab_case_with_odd_expected_fields_does_not_stop_the_run(self):
        coach = CoachService(model(), runners(), self.repo)
        self.fake.chat_handler = smart_handler
        odd = evals.Case("odd", "extract", "Anna", {"races": [["A", 2020]]}, form="runner")     # malformed race
        good = evals.cases("extract", coach)[-1]
        res = evals.run_lab(self.ai, "extract", ["test/a"], [odd, good], coach)
        self.assertEqual(len(res), 2)
        self.assertIsNotNone(res[0].error)

    def test_whole_numbers_only_in_assistant_scores(self):
        case = evals.Case("n", "assistant", "q", {"tool": "predict", "contains": ["5"]})
        step = type("S", (), {"name": "predict", "result": {}})()
        self.assertLess(evals.score_assistant(case, "Based on 15 runners.", [step])[0], 1)
        self.assertEqual(evals.score_assistant(case, "Based on 5 runners.", [step])[0], 1)


class LabTests(unittest.TestCase):
    def setUp(self):
        self.repo = Repository(":memory:")
        self.coach = CoachService(model(), runners(), self.repo)
        self.fake = FakeOpenRouter()
        self.ai = AIService(self.repo, env={}, client=self.fake.client())

    def test_wer(self):
        self.assertEqual(evals.wer("she ran 3:21:04", "she ran 3:21:04"), 0)
        self.assertAlmostEqual(evals.wer("she ran three hours", "he ran three hours today"), 0.5)
        self.assertEqual(evals.wer("", ""), 0)

    def test_every_task_has_scorable_cases(self):
        for task in ("assistant", "extract", "refine"):
            cs = evals.cases(task, self.coach)
            self.assertGreaterEqual(len(cs), 4, task)
        self.assertEqual(evals.cases("transcribe", self.coach), [])

    def test_extract_scores(self):
        case = next(c for c in evals.cases("extract", self.coach) if c.id == "runner-spoken")
        right = {"name": "Anna de Vries", "gender": "F", "age": 34,
                 "races": [{"event": "Amsterdam", "year": 2023, "finish": "3:35:48"},
                           {"event": "Rotterdam marathon", "year": 2024, "finish": "3:33:11"}]}
        self.assertEqual(evals.score_extract(case, right)[0], 1.0)
        wrong = {**right, "age": None, "races": right["races"][:1]}
        score, details = evals.score_extract(case, wrong)
        self.assertLess(score, 0.8)
        self.assertTrue(any("age" in d for d in details))

    def test_assistant_case_grounding(self):
        case = evals.cases("assistant", self.coach)[0]
        finish = case.expected["contains"][0]
        self.fake.chats = [chat_reply(None, [("predict", {"runner_id": case.input.split()[-1].rstrip("?")})]),
                           chat_reply(f"The model predicts {finish}.")]
        r = evals.run_case(self.ai, "assistant", "test/a", case, self.coach)
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.passed)
        self.fake.chats = [chat_reply("It is about 3:10:00, I think.")]           # no function, made-up time
        r = evals.run_case(self.ai, "assistant", "test/b", case, self.coach)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(any("not returned by any function" in d for d in r.details))

    def test_run_lab_summary_and_saving(self):
        self.fake.chat_handler = smart_handler
        cs = evals.cases("extract", self.coach)
        res = evals.run_lab(self.ai, "extract", ["test/a", "test/b"], cs, self.coach)
        self.assertEqual(len(res), 2 * len(cs))
        self.assertEqual({b["model"] for b in self.fake.bodies("/chat/completions")}, {"test/a", "test/b"})
        table = evals.summary(res)
        self.assertEqual(list(table.columns), ["Model", "Score", "Passed", "Errors", "Seconds per case", "Cost (USD)"])
        self.assertEqual(len(table), 2)
        self.fake.chats = [(500, "provider down")]
        bad = evals.run_case(self.ai, "refine", "test/c", evals.cases("refine", self.coach)[0], self.coach)
        self.assertEqual((bad.score, bad.passed), (0.0, False))
        self.assertIn("provider down", bad.error)
        evals.save_results(self.repo, res + [bad])
        saved = evals.saved_results(self.repo)
        self.assertEqual(len(saved), len(res) + 1)

    def test_transcribe_case(self):
        self.fake.transcripts = ["she finished in three hours"]
        case = evals.Case("rec", "transcribe", "(recording)", {"reference": "She finished in three hours."}, audio=b"x")
        r = evals.run_case(self.ai, "transcribe", "openai/whisper-1", case, self.coach)
        self.assertEqual(r.score, 1.0)


if __name__ == "__main__":
    unittest.main()
