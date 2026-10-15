"""Ask PaceSense: input checks, the functions the AI may call, the conversation loop, and the offline helper."""
import datetime as dt
import json
import unittest

from pacesense.assistant import (Assistant, AssistantError, AssistantTools, OfflineModel, OpenRouterModel,
                                 check_runner, parse_duration)
from pacesense.assistant.offline import parse_runner_text
from pacesense.assistant.tools import SPECS
from pacesense.coach import CoachService
from pacesense.planning import PlanSession
from pacesense.storage import Repository

from .fake_openrouter import FakeOpenRouter, chat_reply
from .helpers import model, runners

TODAY = dt.date(2026, 10, 10)


class IntakeTests(unittest.TestCase):
    def test_parse_duration(self):
        for text, secs in [("3:35:48", 12948), ("3:35", 12900), ("3h35m48s", 12948), ("3h35", 12900),
                           ("3 hours 35 min", 12900), ("215 min", 12900), ("215", 12900), (12948, 12948),
                           ("3,5h", 12600)]:
            self.assertEqual(parse_duration(text), secs, text)
        for bad in ("fast", "", None, "3:75", "1:2:3:4"):
            self.assertIsNone(parse_duration(bad), bad)

    def test_clean_runner_with_corrections(self):
        r = check_runner({"name": "  Anna   de Vries ", "gender": "woman", "age": "34",
                          "races": [{"event": "Rotterdam", "year": 2024, "finish": "3h33m11"},
                                    {"event": "Amsterdam", "year": 2023, "finish": "3:35:48"}]}, TODAY)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["runner"]["name"], "Anna de Vries")
        self.assertEqual(r["runner"]["gender"], "F")
        self.assertEqual([x["year"] for x in r["runner"]["races"]], [2023, 2024])      # oldest first
        self.assertEqual(r["runner"]["races"][1]["finish"], "3:33:11")
        self.assertTrue(any("3h33m11" in c for c in r["corrections"]))

    def test_problems_and_missing(self):
        r = check_runner({"name": "", "gender": "x", "age": 140,
                          "races": [{"event": "Berlin", "year": 2030, "finish": "9:10:00"},
                                    {"event": "Berlin", "year": 2030, "finish": "3:10:00"},
                                    {"event": "Paris", "finish": ""}]}, TODAY)
        self.assertFalse(r["ok"])
        text = " ".join(r["problems"] + r["missing"])
        for expected in ("name", "not recognised", "outside 10 to 100", "not between 1980 and 2026",
                         "is it a marathon time", "appears twice", "year of", "finish time of"):
            self.assertIn(expected, text)

    def test_section_splits_are_added_up(self):
        sections = ["24:10", "24:05", "24:20", "24:30", "24:40", "25:00", "25:30", "26:00"]
        r = check_runner({"name": "X", "gender": "F", "races": [{"event": "A", "year": 2020, "finish": "3:35:00",
                                                                 "splits": sections}]}, TODAY)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["runner"]["races"][0]["splits"][-1], "3:18:15")
        self.assertTrue(any("added up" in c for c in r["corrections"]))

    def test_bad_splits(self):
        base = {"name": "X", "gender": "M"}
        few = check_runner({**base, "races": [{"event": "A", "year": 2020, "finish": "3:30:00", "splits": "0:25:00"}]}, TODAY)
        self.assertIn("8 (at 5", " ".join(few["problems"]))
        late = ["0:25:00", "0:50:00", "1:15:00", "1:40:00", "2:05:00", "2:30:00", "2:55:00", "3:40:00"]
        r = check_runner({**base, "races": [{"event": "A", "year": 2020, "finish": "3:30:00", "splits": late}]}, TODAY)
        self.assertIn("not before the finish", " ".join(r["problems"]))
        jump = ["0:25:00", "0:50:00", "0:55:00", "1:40:00", "2:05:00", "2:30:00", "2:55:00", "3:20:00"]
        r = check_runner({**base, "races": [{"event": "A", "year": 2020, "finish": "3:30:00", "splits": jump}]}, TODAY)
        self.assertIn("not a realistic marathon pace", " ".join(r["problems"]))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.coach = CoachService(model(), runners(), Repository(":memory:"))
        self.rid = next(r.runner_id for r in runners() if len(r.races) >= 3)
        self.plan = PlanSession(self.coach, self.rid)
        self.tools = AssistantTools(self.coach, plan=lambda: self.plan)

    def test_specs_are_valid_and_cover_every_function(self):
        names = [s["function"]["name"] for s in SPECS]
        self.assertEqual(len(names), len(set(names)))
        for n in names:
            self.assertTrue(hasattr(self.tools, f"_t_{n}"), n)
        self.assertFalse(any("study" in n or "session" in n for n in names))    # no access to study data
        json.dumps(SPECS)

    def test_predict_matches_the_dashboard(self):
        r = self.tools.call("predict", {"runner_id": self.rid})
        self.assertEqual(r["finish"], self.coach.predict(self.rid)["finish"])
        self.assertIn(" to ", r["likely_range"])
        n = len(self.plan.weights)
        w = [0.1] + [1.0] * (n - 1)
        r2 = self.tools.call("predict", {"runner_id": self.rid, "weights": w})
        self.assertEqual(list(r2["weights_used"].values()), w)
        self.assertIn("error", self.tools.call("predict", {"runner_id": self.rid, "weights": [0.5]}))
        self.assertIn("error", self.tools.call("predict", {"runner_id": "nobody"}))
        json.dumps(r)

    def test_current_plan_and_changes(self):
        self.plan.set_weight(0, 0.1)
        r = self.tools.call("current_plan", {})
        self.assertEqual(r["runner_id"], self.rid)
        self.assertIn("weight changed from 0.5 to 0.1", r["changes_newest_first"][0])
        none = AssistantTools(self.coach).call("current_plan", {})
        self.assertIsNone(none["open_plan"])

    def test_check_pacing_plan(self):
        paces = [f"{int(p // 60)}:{int(p % 60):02d}" for p in self.plan.paces]
        r = self.tools.call("check_pacing_plan", {"runner_id": self.rid, "paces": paces})
        self.assertIn(r["verdict"], ("realistic", "ambitious", "cautious"))
        self.assertIn("error", self.tools.call("check_pacing_plan", {"runner_id": self.rid, "paces": paces[:3]}))
        self.assertIn("error", self.tools.call("check_pacing_plan", {"runner_id": self.rid, "paces": ["fast"] * 9}))

    def test_watch_and_saved_plans(self):
        r = self.tools.call("watch_moment", {"distance_km": 21.1, "elapsed": "1:40:00"})
        self.assertIn("watch_cue", r)
        saved = self.plan.save("Toronto 2026")
        self.assertEqual(self.tools.call("saved_plans", {})["plans"][0]["plan_id"], saved["id"])
        r = self.tools.call("watch_moment", {"distance_km": 10, "elapsed": "0:50:00", "plan_id": saved["id"]})
        self.assertIn("Toronto 2026", r["plan"])
        self.assertIn("error", self.tools.call("watch_moment", {"distance_km": 10, "elapsed": "x"}))

    def test_bad_calls_return_errors(self):
        self.assertIn("error", self.tools.call("delete_everything", {}))
        self.assertIn("error", self.tools.call("predict", "{not json"))
        self.assertIn("error", self.tools.call("predict", {"wrong": 1}))
        self.assertIn("error", self.tools.call("explain", {"topic": "nonsense"}))

    def test_propose_runner_only_when_clean(self):
        bad = self.tools.call("propose_runner", {"name": "X", "gender": "F", "races": [{"event": "A", "finish": "x"}]})
        self.assertFalse(bad["proposed"])
        self.assertIsNone(self.tools.pending_runner)
        ok = self.tools.call("propose_runner", {"name": "X", "gender": "F",
                                                "races": [{"event": "A", "year": 2020, "finish": "3:30"}]})
        self.assertTrue(ok["proposed"])
        self.assertEqual(self.tools.pending_runner["races"][0]["finish"], "3:30:00")
        self.assertEqual(self.coach.search("X"), [])                              # nothing saved


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.coach = CoachService(model(), runners(), Repository(":memory:"))
        self.rid = runners()[3].runner_id
        self.tools = AssistantTools(self.coach)
        self.fake = FakeOpenRouter()

    def test_openrouter_loop_runs_functions_and_answers(self):
        self.fake.chats = [chat_reply(None, [("predict", {"runner_id": self.rid})]), chat_reply("The target is ...")]
        m = OpenRouterModel(self.fake.client(), "test/model", 0.1)
        ans = Assistant(m, self.tools).ask("Predict please", [{"role": "user", "content": "hi"},
                                                              {"role": "assistant", "content": "hello"}])
        self.assertEqual(ans.text, "The target is ...")
        self.assertEqual([s.name for s in ans.steps], ["predict"])
        self.assertAlmostEqual(ans.usage.cost, 0.0002)
        first, second = self.fake.bodies("/chat/completions")
        self.assertEqual((first["model"], first["temperature"]), ("test/model", 0.1))
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertEqual([m["role"] for m in first["messages"][1:]], ["user", "assistant", "user"])
        self.assertEqual(len(first["tools"]), len(SPECS))
        tool_msg = second["messages"][-1]
        self.assertEqual((tool_msg["role"], tool_msg["tool_call_id"]), ("tool", "call_0"))
        self.assertEqual(json.loads(tool_msg["content"])["finish"], self.coach.predict(self.rid)["finish"])

    def test_loop_stops_after_max_rounds(self):
        self.fake.chats = [chat_reply(None, [("saved_plans", {})])] * 3
        ans = Assistant(OpenRouterModel(self.fake.client(), "m/x"), self.tools, max_rounds=3).ask("loop")
        self.assertIn("too many steps", ans.text)
        self.assertEqual(len(ans.steps), 3)

    def test_service_errors_become_assistant_errors(self):
        self.fake.chats = [(401, "No auth credentials found")]
        with self.assertRaises(AssistantError) as e:
            Assistant(OpenRouterModel(self.fake.client(), "m/x"), self.tools).ask("hi")
        self.assertIn("No auth credentials", str(e.exception))


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.coach = CoachService(model(), runners(), Repository(":memory:"))
        self.rid = next(r.runner_id for r in runners() if len(r.races) >= 3)
        self.plan = PlanSession(self.coach, self.rid)
        self.tools = AssistantTools(self.coach, plan=lambda: self.plan)
        self.a = Assistant(OfflineModel(), self.tools)

    def steps(self, q):
        ans = self.a.ask(q)
        return [s.name for s in ans.steps], ans.text

    def test_routes(self):
        cases = {f"Predict {self.rid}": ["predict"], "Is my plan realistic?": ["current_plan"],
                 "What is the likely range?": ["predict", "explain"], "Explain the race weights": ["explain"],
                 "At 21.1 km in 1:41:00 am I on plan?": ["watch_moment"], "Show my saved plans": ["saved_plans"],
                 "Find runner R0001": ["find_runners"], f"Past races of {self.rid}": ["runner_details"]}
        for q, expected in cases.items():
            self.assertEqual(self.steps(q)[0], expected, q)
        names, text = self.steps("hello")
        self.assertEqual(names, [])
        self.assertIn("offline helper", text)

    def test_numbers_come_from_functions(self):
        _, text = self.steps(f"Predict {self.rid}")
        self.assertIn(self.coach.predict(self.rid)["finish"], text)

    def test_add_runner_conversation(self):
        names, text = self.steps("Add runner Anna de Vries, female, 34: Amsterdam 2023 3h35m48; Rotterdam 2024 3:33:11")
        self.assertEqual(names, ["check_runner_details", "propose_runner"])
        self.assertIn("Add this runner", text)
        self.assertEqual(self.tools.pending_runner["races"][0]["finish"], "3:35:48")
        names, text = self.steps("Add runner Bob, man: Berlin 2030 9:10:00")
        self.assertEqual(names, ["check_runner_details"])
        self.assertIn("Problem", text)

    def test_parse_runner_text(self):
        d = parse_runner_text("add a new runner Sam Lee, f, age 41, ran Toronto 2022 in 4:01:10 and Boston 2024 3:58")
        self.assertEqual((d["name"], d["gender"], d["age"]), ("Sam Lee", "f", 41))
        self.assertEqual([(r["event"], r["year"], r["finish"]) for r in d["races"]],
                         [("Toronto", 2022, "4:01:10"), ("Boston", 2024, "3:58")])


if __name__ == "__main__":
    unittest.main()
