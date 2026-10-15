"""Runs every Streamlit page against the stand-in in fake_streamlit.py, clicking through real workflows."""
import importlib
import os
import sys
import unittest

from . import fake_streamlit as fake

sys.modules["streamlit"] = fake                    # must happen before the dashboard is imported

from pacesense.coach import CoachService           # noqa: E402
from pacesense.dashboard import services as svc    # noqa: E402
from pacesense.dashboard import ui                 # noqa: E402
from pacesense.dashboard.views import (assistant, howitworks, lab, plan, record, results, runners, saved, settings,  # noqa: E402
                                       study, watch)
from pacesense.llm.service import AIService         # noqa: E402
from pacesense.storage import Repository           # noqa: E402
from pacesense.study import StudyService           # noqa: E402

from .fake_openrouter import FakeOpenRouter, smart_handler  # noqa: E402
from .helpers import model, runners as sample_runners  # noqa: E402


def make_services(assignment="random", token="secret"):
    repo = Repository(":memory:")
    return svc.Services(CoachService(model(), sample_runners(), repo),
                        StudyService(model(), sample_runners(), repo, data_source="test", seed=3),
                        repo, "test data", token, assignment)


class PageTest(unittest.TestCase):
    def setUp(self):
        fake.reset()
        self.s = make_services()
        ui.services = lambda: self.s
        ui.ai = lambda: AIService(self.s.repo, env={})                       # offline unless a test says otherwise
        ui.PAGES.update({k: fake.Page(lambda: None, url_path=k) for k in
                         ("lab", "plan", "watch", "ask", "runners", "saved", "record", "study", "results", "help", "settings")})

    def online(self, transcripts=()):
        """Use a fake OpenRouter for the AI features."""
        self.router = FakeOpenRouter()
        self.router.chat_handler = smart_handler
        self.router.transcripts = list(transcripts)
        self.s.repo.set_setting("transcription_backend", "openrouter")
        ui.ai = lambda: AIService(self.s.repo, env={}, client=self.router.client())
        return self.router

    def demo_plan(self):
        fake.click("welcome_new")
        fake.run(plan.render)
        fake.click("runner_demo")
        fake.run(plan.render)
        return ui.plan()


class PlanPageTests(PageTest):
    def test_guided_flow_end_to_end(self):
        fake.run(plan.render)
        self.assertIn("How would you like to use PaceSense?", fake.texts())
        p = self.demo_plan()
        self.assertIsNotNone(p)
        self.assertEqual(self.s.coach.preferences()["experience"], "new")
        self.assertEqual(fake.session_state["step"], 2)
        self.assertIn("Weight the past races", fake.texts())
        self.assertIn("No changes yet", fake.texts())

        key = f"weight_{p.runner_id}_0_{p.version}"
        fake.interact(key, 0.1)
        fake.run(plan.render)
        self.assertIn("weight changed from 0.5 to 0.1", fake.texts())          # summary bar explains it
        self.assertTrue(any("weight changed" in t for t in fake.kinds("toast")))     # queued, then shown

        fake.click("step_next"); fake.run(plan.render)
        self.assertEqual(fake.session_state["step"], 3)
        self.assertIn("About 8 in 10 runners like", fake.texts())

        fake.click("step_next"); fake.run(plan.render)
        self.assertEqual(fake.session_state["step"], 4)
        fake.interact(f"strategy_{p.runner_id}_{p.version}", "negative")
        self.assertEqual(p.source, "negative")
        fake.run(plan.render)
        editor_key = f"paces_{p.runner_id}_{p.version}"
        table = fake.kinds("data_editor")[0]
        edited = table.copy()
        col = [c for c in edited.columns if c.startswith("Pace")][0]
        edited.loc[0, col] = "4:00"
        fake.set_value(editor_key, edited)
        fake.run(plan.render)
        self.assertEqual(p.source, "custom")
        self.assertIn("Pace for 0–5 km changed", p.changes[0].text())

        fake.click("step_next"); fake.run(plan.render)
        self.assertEqual(fake.session_state["step"], 5)
        fake.click("plan_save"); fake.run(plan.render)
        self.assertIn("Enter the target race before saving", fake.texts())
        fake.session_state["target_race"] = "Toronto Waterfront Marathon 2026"
        fake.click("plan_save"); fake.run(plan.render)
        self.assertIn("Plan saved", fake.texts())
        self.assertEqual(len(self.s.coach.list_plans()), 1)
        self.assertIn(f"{p.name}-plan.txt".replace(" ", "_"), fake.kinds("download"))
        self.assertGreaterEqual(len([t for k, t in fake.A.out if k == "markdown" and ". " in t and "changed" in t]), 3)

        fake.click("step_back"); fake.run(plan.render)
        self.assertEqual(fake.session_state["step"], 4)

    def test_talk_to_the_sliders(self):
        p = self.demo_plan()
        year = p.overview["races"][0]["year"]
        fake.interact("talk_text", f"{year} was a hot day")
        self.assertEqual(p.weights[0], 0.2)
        self.assertIn('You said: "hot day"', p.changes[0].text())
        fake.run(plan.render)
        self.assertIn(f"{year} was a hot day", fake.texts())
        fake.interact("talk_text", "I think she can run 2:45")
        fake.run(plan.render)
        self.assertIn("Steering index", fake.texts())
        fake.interact("talk_text", "scratch that")
        self.assertEqual(p.weights[0], 0.5)

    def test_all_controls_layout(self):
        self.s.coach.update_preferences({"experience": "experienced", "view": "full"})
        fake.click("runner_demo"); fake.run(plan.render)
        fake.run(plan.render)
        out = fake.texts()
        for heading in ("Weight the past races", "Shape the pacing", "Check and save", "Model prediction"):
            self.assertIn(heading, out)
        self.assertFalse(any(label.startswith("1. Runner") for k, label in fake.A.out if k == "button"))

    def test_layout_switch_and_reset(self):
        p = self.demo_plan()
        fake.interact(f"weight_{p.runner_id}_1_{p.version}", 0.9)
        fake.click("weights_reset"); fake.run(plan.render)
        self.assertTrue(all(w == 0.5 for w in p.weights))
        fake.set_value("layout_guided", "All controls")
        fake.run(plan.render)
        self.assertEqual(self.s.coach.preferences()["view"], "full")


class WatchPageTests(PageTest):
    def test_no_plan(self):
        fake.run(watch.render)
        self.assertIn("There is no plan yet", fake.texts())

    def test_moment_and_simulation(self):
        p = self.demo_plan()
        p.save("Toronto 2026")
        fake.run(watch.render)
        self.assertTrue(fake.kinds("pyplot"))                                # the watch face
        self.assertIn("Target pace", fake.texts())
        fake.session_state["watch_offset"] = 60
        fake.run(watch.render)
        self.assertIn("behind", fake.texts())
        self.assertIn("paceband.txt", " ".join(fake.kinds("download")))

        fake.session_state["watch_mode"] = "Simulate a race"
        fake.run(watch.render)
        self.assertEqual(fake.session_state["sim"]["clock"], 0.0)
        fake.click("sim_toggle"); fake.run(watch.render)
        self.assertTrue(fake.session_state["sim"]["running"])
        before = fake.session_state["sim"]["clock"]
        fake.run(watch.render)                                               # one timer tick
        self.assertGreater(fake.session_state["sim"]["clock"], before)
        fake.click("sim_jump"); fake.run(watch.render)
        self.assertGreater(fake.session_state["sim"]["clock"], 2 * 3600 - 1800)
        for _ in range(400):                                                 # run to the finish
            if not fake.session_state["sim"]["running"]:
                break
            fake.run(watch.render)
        self.assertFalse(fake.session_state["sim"]["running"])
        self.assertIn("Race finished", fake.texts())


class OtherPageTests(PageTest):
    def test_runners_search_and_add(self):
        fake.session_state["runners_query"] = "R000"
        fake.run(runners.render)
        self.assertTrue(fake.kinds("dataframe"))
        fake.click("runners_plan")
        self.assertIsNotNone(fake.run(runners.render))                       # switched to Plan a race
        self.assertIsNotNone(ui.plan())

        fake.reset(); ui.services = lambda: self.s
        df = runners.BLANK_RACES.copy()
        df.loc[0] = ["Amsterdam", 2023, "3:35:48", ""]
        df.loc[1] = ["Rotterdam", 2024, "3:33:11", ""]
        fake.set_value("add_runner_races", df)
        fake.set_value("Name", "Anna de Vries")
        fake.click("Add runner")
        self.assertIsNotNone(fake.run(runners.render))
        self.assertEqual(ui.plan().name, "Anna de Vries")

        fake.reset(); ui.services = lambda: self.s
        bad = runners.BLANK_RACES.copy()
        bad.loc[0] = ["Amsterdam", 2023, "fast", ""]
        fake.set_value("add_runner_races", bad)
        fake.set_value("Name", "X")
        fake.click("Add runner")
        self.assertIsNone(fake.run(runners.render))
        self.assertIn("finish must be a time", fake.texts())

    def test_saved_and_record(self):
        fake.run(saved.render)
        self.assertIn("No saved plans yet", fake.texts())
        p = self.demo_plan()
        p.save("Toronto 2026", "2026-10-18", "note")
        fake.run(saved.render)
        fake.set_value("Actual finish time (h:mm:ss)", "3:20:00")
        fake.click("Record result")
        fake.run(saved.render)
        self.assertIn("Recorded. The plan was off by", fake.texts())
        fake.click("saved_open")
        self.assertIsNotNone(fake.run(saved.render))
        self.assertEqual(fake.session_state["target_race"], "Toronto 2026")
        fake.run(record.render)
        self.assertIn("Races with a result: 1", fake.texts())

    def test_settings_and_help(self):
        p = self.demo_plan()
        fake.set_value("set_unit_km", "mi")
        fake.run(settings.render)
        self.assertEqual(self.s.coach.preferences()["unit"], "mi")
        self.assertEqual(p.unit, "mi")                                        # the open plan follows
        fake.click("set_welcome")
        self.assertIsNotNone(fake.run(settings.render))
        self.assertIsNone(self.s.coach.preferences()["experience"])
        fake.run(howitworks.render)
        self.assertIn("How the code is organised", fake.texts())

    def test_results_needs_token(self):
        fake.run(results.render)
        fake.session_state["admin_token"] = "wrong"
        fake.click("admin_unlock"); fake.run(results.render)
        self.assertIn("That token is not right", fake.texts())
        fake.session_state["admin_token"] = "secret"
        fake.click("admin_unlock"); fake.run(results.render)
        self.assertIn("pacesense_trials.csv", fake.kinds("download"))
        self.s = make_services(token=None)
        fake.reset(); ui.services = lambda: self.s
        fake.run(results.render)
        self.assertIn("This page is turned off", fake.texts())


class StudyPageTests(PageTest):
    def complete_study(self, assignment):
        self.s = make_services(assignment=assignment)
        ui.services = lambda: self.s
        fake.run(study.render)
        self.assertIn("Informed consent", fake.texts())
        fake.session_state["study_consent"] = True
        fake.click("study_start"); fake.run(study.render)
        fake.click("study_skip"); fake.run(study.render)
        sid = fake.session_state["study"]["sid"]
        for seq in range(1, 5):
            self.assertIn(f"Runner {seq} of 4", fake.texts())
            fake.set_value("What do you find a challenging yet realistic finish time for this runner's next "
                           "marathon? (h:mm:ss)", "3:40:00")
            fake.click("Next")
            fake.run(study.render)
            self.assertIn("Model prediction", fake.texts())
            if assignment == "interactive":
                n = len(self.s.study.trial_view(sid, seq)["races"])
                fake.interact(f"study_w_{sid}_{seq}_0", 0.1 if n > 1 else 1.0)
            fake.set_value("New finish time: now you have assessed the model, what do you find a challenging yet "
                           "realistic finish time for this runner? (h:mm:ss)", "3:38:00")
            fake.click({1: "Go to the next runner", 2: "Go to the next runner", 3: "Go to the last runner",
                        4: "Go to the questionnaire"}[seq])
            fake.run(study.render)
            for k in [k for k in list(fake.A.values) if "finish time" in k]:
                del fake.A.values[k]
        self.assertIn("Questionnaire", fake.texts())
        fake.click("Submit"); fake.run(study.render)
        self.assertIn("Please answer all statements", fake.texts())
        q = self.s.study.questionnaire_for(sid)
        for scale in q["scales"]:
            for item in scale["items"]:
                fake.session_state[f"study_q_{item['id']}"] = 5
        fake.click("Submit"); fake.run(study.render)
        self.assertIn("Thank you", fake.texts())
        return sid

    def test_noninteractive_study(self):
        sid = self.complete_study("noninteractive")
        trials = self.s.repo.trials_frame()
        self.assertEqual(len(trials), 4)
        self.assertTrue((trials["interactions"] == 0).all())
        self.assertTrue(self.s.study.session_summary(sid)["completed"])

    def test_interactive_study_records_slider_use(self):
        self.complete_study("interactive")
        trials = self.s.repo.trials_frame()
        self.assertGreater(trials["interactions"].sum(), 0)


class AssistantPageTests(PageTest):
    def setUp(self):
        super().setUp()
        self._env = {k: os.environ.pop(k) for k in ("OPENAI_API_KEY", "PACESENSE_ASSISTANT") if k in os.environ}

    def tearDown(self):
        os.environ.update(self._env)

    def test_suggestions_questions_and_trace(self):
        fake.run(assistant.render)
        self.assertIn("Offline helper", fake.texts())
        self.assertIn("Show my saved plans", [l for k, l in fake.A.out if k == "button"])
        p = self.demo_plan()
        fake.run(assistant.render)
        fake.click("assistant_suggest_0")                      # "Is my plan for … realistic?"
        fake.run(assistant.render)
        chat = fake.session_state["chat"]
        self.assertEqual([m["role"] for m in chat], ["user", "assistant"])
        self.assertIn(p.name, chat[1]["content"])
        self.assertIn("What PaceSense calculated (1)", fake.texts())
        self.assertTrue(fake.kinds("json"))
        fake.set_value("assistant_input", "Explain the race weights")
        fake.run(assistant.render)
        self.assertIn("Race weights", fake.session_state["chat"][-1]["content"])
        fake.click("assistant_clear"); fake.run(assistant.render)
        self.assertEqual(fake.session_state["chat"], [])

    def test_add_runner_from_chat(self):
        fake.set_value("assistant_input", "Add runner Anna de Vries, female, 34: Amsterdam 2023 3:35:48; "
                                          "Rotterdam 2024 3h33m11")
        fake.run(assistant.render)
        self.assertIn("Ready to add: Anna de Vries", fake.texts())
        self.assertEqual(self.s.coach.search("Anna"), [])                # not saved until the coach confirms
        fake.click("assistant_add")
        self.assertIsNotNone(fake.run(assistant.render))                  # went to Plan a race
        self.assertEqual(ui.plan().name, "Anna de Vries")
        self.assertIsNone(fake.session_state.get("assistant_pending"))

    def test_service_error_is_shown(self):
        router = self.online()
        router.chats = [(401, "No auth credentials found")]
        fake.set_value("assistant_input", "hi")
        fake.run(assistant.render)
        self.assertIn("did not answer: No auth credentials", fake.session_state["chat"][-1]["content"])

    def test_voice_question_refine_and_send(self):
        fake.run(assistant.render)
        self.assertIn("Record", fake.texts())  # local transcription remains available without an OpenRouter key
        router = self.online(["um is my plan for this runner realistic"])
        fake.set_value("ask_voice_rec", fake.UploadedFile(b"RIFF...."))
        fake.run(assistant.render)
        self.assertEqual(fake.session_state["ask_draft"], "um is my plan for this runner realistic")
        self.assertIn("Suggested clearer version", fake.texts())
        fake.click("ask_voice_use_refined")
        fake.run(assistant.render)
        self.assertEqual(fake.session_state["ask_draft"], "Um is my plan for this runner realistic.")
        router.chats = []
        fake.click("ask_send")
        fake.run(assistant.render)
        self.assertEqual(fake.session_state["chat"][0]["content"], "Um is my plan for this runner realistic.")
        meta = fake.session_state["chat"][1]["meta"]                                # who answered, and the cost
        self.assertIn("openai/gpt-4o-mini via OpenRouter", meta)
        self.assertIn("$0.0001", meta)
        self.assertEqual(fake.session_state["ask_draft"], "")
        fake.run(assistant.render)                          # the recording is still in the widget: not sent again
        self.assertEqual(len(router.bodies("/audio/transcriptions")), 1)
        self.assertNotIn("Suggested clearer version", fake.texts())

    def test_refine_typed_text(self):
        self.online()
        fake.session_state["ask_draft"] = "what range for the runner"
        fake.click("ask_refine_ask")
        fake.run(assistant.render)
        self.assertIn("What range for the runner.", fake.texts())
        fake.click("ask_refine_use_refined")
        fake.run(assistant.render)
        self.assertEqual(fake.session_state["ask_draft"], "What range for the runner.")


class VoiceFormTests(PageTest):
    def test_runner_by_voice_fills_the_form(self):
        self.online(["new runner anna de vries female 34 amsterdam 2023 3:35:48 rotterdam 2024 3h33m11"])
        fake.set_value("runner_voice_rec", fake.UploadedFile(b"RIFF"))
        fake.run(runners.render)
        self.assertEqual((fake.session_state["new_name"], fake.session_state["new_gender"],
                          fake.session_state["new_age"]), ("Anna de Vries", "F", 34))
        races = fake.session_state["new_races"]
        self.assertEqual(list(races["Finish (h:mm:ss)"]), ["3:35:48", "3:33:11"])
        self.assertIn("Corrected: Rotterdam 2024: read finish '3h33m11' as 3:33:11.", fake.texts())
        self.assertEqual(len(self.router.bodies("/audio/transcriptions")), 1)
        fake.run(runners.render)                                                   # same recording: not sent again
        self.assertEqual(len(self.router.bodies("/audio/transcriptions")), 1)

    def test_out_of_range_age_is_not_put_in_the_age_box(self):
        runners.apply_runner({"name": "X", "age": 140, "races": []}, "")
        self.assertNotIn("new_age", fake.session_state)
        self.assertEqual(fake.session_state["new_name"], "X")

    def test_transcription_error_offers_retry(self):
        router = self.online()
        router.transcripts = [(500, "upstream timeout"), "three twenty one oh four"]
        p = self.demo_plan()
        sp = p.save("Toronto 2026")
        key = f"result_voice_{sp['id']}"
        fake.set_value(f"{key}_rec", fake.UploadedFile(b"RIFF"))
        fake.run(saved.render)
        self.assertIn("Could not transcribe the recording: upstream timeout", fake.texts())
        fake.run(saved.render)                                                     # not retried by itself
        self.assertEqual(len(router.bodies("/audio/transcriptions")), 1)
        fake.click(f"{key}_retry")
        fake.run(saved.render)          # the stand-in runs callbacks where the button is drawn; Streamlit runs them
        fake.run(saved.render)          # before the next run, so one more run here
        self.assertEqual(fake.session_state[f"result_{sp['id']}"], "3:21:04")

    def test_widget_values_survive_leaving_the_step(self):
        fake.session_state["target_race"] = "Toronto"
        ui.keep_widget_values()
        self.assertEqual(fake.session_state["target_race"], "Toronto")

    def test_plan_details_and_result_by_voice(self):
        p = self.demo_plan()
        self.online(["the target is toronto waterfront on october 18 2026 calf strain in august"])
        fake.session_state["step"] = 5
        fake.set_value("plan_voice_rec", fake.UploadedFile(b"RIFF"))
        fake.run(plan.render)
        import datetime as dt
        self.assertEqual(fake.session_state["target_race"], "Toronto Waterfront Marathon")
        self.assertEqual(fake.session_state["target_date"], dt.date(2026, 10, 18))
        self.assertIn("Calf strain", fake.session_state["target_notes"])
        saved_plan = p.save("Toronto 2026")
        fake.reset(); ui.services = lambda: self.s
        self.online(["three twenty one oh four"])
        fake.set_value(f"result_voice_{saved_plan['id']}_rec", fake.UploadedFile(b"RIFF"))
        fake.run(saved.render)
        self.assertEqual(fake.session_state[f"result_{saved_plan['id']}"], "3:21:04")

    def test_upload_instead_of_recording(self):
        router = self.online(["three twenty one oh four"])
        p = self.demo_plan()
        sp = p.save("Toronto 2026")
        fake.set_value(f"result_voice_{sp['id']}_file", fake.UploadedFile(b"ID3", "finish.mp3"))
        fake.run(saved.render)
        self.assertEqual(router.bodies("/audio/transcriptions")[0]["input_audio"]["format"], "mp3")


class AISettingsAndLabTests(PageTest):
    def test_model_per_task_in_settings(self):
        fake.run(settings.render)
        self.assertIn("AI models and voice input", fake.texts())
        self.assertIn("No OpenRouter key", fake.texts())
        cur = AIService(self.s.repo, env={}).settings.get("refine")
        fake.set_value(f"ai_model_refine_{cur['model']}", "google/gemini-2.5-flash")
        fake.run(settings.render)
        self.assertEqual(AIService(self.s.repo, env={}).settings.get("refine")["model"], "google/gemini-2.5-flash")
        fake.set_value(f"ai_model_extract_{cur['model']}", "not a model id")
        fake.run(settings.render)
        self.assertIn("looks like provider/model", fake.texts())
        fake.interact("ai_key_input", "sk-or-typed")                              # pasted key, kept for the session
        self.assertEqual(fake.session_state["openrouter_key"], "sk-or-typed")
        self.online()
        fake.click("ai_load_models")
        fake.run(settings.render)
        self.assertIn("Showing 2 of 2 matching models", fake.texts())

    def test_lab_runs_compares_and_chooses(self):
        fake.run(lab.render)
        self.assertIn("needs an OpenRouter key", fake.texts())
        router = self.online()
        fake.set_value("lab_task", "extract")
        fake.session_state["lab_models_extract"] = "test/a\ntest/b"
        fake.run(lab.render)
        fake.click("lab_run")
        fake.run(lab.render)
        self.assertEqual({b["model"] for b in router.bodies("/chat/completions")}, {"test/a", "test/b"})
        tables = fake.kinds("dataframe")
        self.assertEqual(set(tables[0]["Model"]), {"test/a", "test/b"})               # the summary, one row per model
        self.assertIn("Score", tables[0].columns)
        fake.set_value("lab_pick", "test/b")
        fake.click("lab_use")
        fake.run(lab.render)
        self.assertEqual(AIService(self.s.repo, env={}).settings.get("extract")["model"], "test/b")
        self.assertIn("pacesense_model_lab.csv", fake.kinds("download"))


class AppTests(PageTest):
    def test_app_runs_default_page(self):
        sys.modules.pop("pacesense.dashboard.app", None)
        importlib.import_module("pacesense.dashboard.app")      # runs main(): navigation -> Plan a race
        self.assertIn("Plan a race", fake.texts())
        self.assertEqual(set(ui.PAGES), {"plan", "watch", "ask", "runners", "saved", "record", "study", "results",
                                         "lab", "help", "settings"})


if __name__ == "__main__":
    unittest.main()
