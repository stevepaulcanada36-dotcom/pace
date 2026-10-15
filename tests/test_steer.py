"""PaceSense Live: interpreter, statistics, spoken sessions, benchmark and the checked AI reader."""
import json
import math
import unittest

from pacesense.coach import CoachService
from pacesense.planning import PlanSession
from pacesense.steer import SteeringSession, interpret, race_evidence, recency_weights, replay, steering_index
from pacesense.steer.benchmark import run as run_benchmark
from pacesense.steer.evidence import check_change, influence
from pacesense.steer.llm_interpret import interpret_hybrid, validate_proposal
from pacesense.storage import Repository

from tests import helpers
from tests.fake_openrouter import FakeOpenRouter, chat_reply

RACES = [{"event": "Rotterdam", "year": 2012}, {"event": "Eindhoven", "year": 2014},
         {"event": "Amsterdam", "year": 2016}, {"event": "Rotterdam", "year": 2018}]


def moves(text, context=None):
    it = interpret(text, RACES, context)
    return {RACES[i]["year"]: (a.kind, a.value) for a in it.actions for i in a.races if a.kind in ("set", "shift")}, it


class InterpretTest(unittest.TestCase):
    def test_description_after_the_race(self):
        m, _ = moves("2016 was a hot day and she was injured")
        self.assertEqual(m, {2016: ("shift", -0.6)})

    def test_command_before_the_race(self):
        m, _ = moves("2014 was fine, ignore 2016")
        self.assertEqual(m, {2016: ("set", 0.0)})

    def test_event_names_and_years_in_words(self):
        self.assertEqual(moves("twenty sixteen doesn't count")[0], {2016: ("set", 0.0)})
        self.assertEqual(moves("put Eindhoven at point eight")[0], {2014: ("set", 0.8)})
        self.assertEqual(moves("Rotterdam 2018 is exactly how she runs now")[0], {2018: ("set", 1.0)})

    def test_explicit_number_is_not_cut_at_the_decimal_point(self):
        self.assertEqual(moves("set 2014 to 0.3")[0], {2014: ("set", 0.3)})

    def test_modifiers_and_negation(self):
        self.assertEqual(moves("the last race was slightly windy")[0], {2018: ("shift", -0.1)})
        self.assertEqual(moves("she wasn't injured in 2014")[0], {2014: ("shift", 0.15)})

    def test_context_carries_to_the_next_sentence(self):
        m, it = moves("Amsterdam was terrible. She had the flu.")
        self.assertEqual(m, {2016: ("shift", -0.6)})
        self.assertEqual(it.context, 2)
        self.assertEqual(moves("she was injured", context=2)[0], {2016: ("shift", -0.3)})

    def test_nothing_is_guessed(self):
        m, it = moves("2015 was hot")
        self.assertEqual(m, {})
        self.assertEqual(it.unknown_races, ["2015"])
        m, it = moves("she was injured")            # no race and no context
        self.assertEqual(m, {})
        self.assertEqual(it.unresolved, ["injured"])

    def test_commands_and_estimate(self):
        self.assertEqual([a.kind for a in interpret("trust the recent races more", RACES).actions], ["recency"])
        self.assertEqual([a.kind for a in interpret("reset", RACES).actions], ["reset"])
        self.assertEqual([a.kind for a in interpret("scratch that", RACES).actions], ["undo"])
        self.assertEqual(interpret("I think she can run 3:25", RACES).estimate_s, 3 * 3600 + 25 * 60)
        self.assertEqual(interpret("I reckon 3 hours 22", RACES).estimate_s, 3 * 3600 + 22 * 60)

    def test_deterministic(self):
        a = interpret("2018 was her best race, 2012 was a long time ago", RACES)
        b = interpret("2018 was her best race, 2012 was a long time ago", RACES)
        self.assertEqual(a.actions, b.actions)

    def test_recency_weights(self):
        self.assertEqual(recency_weights([2012, 2014, 2016, 2018], 3.0), [0.2, 0.4, 0.6, 1.0])
        self.assertEqual(recency_weights([2012, 2018], math.inf), [0.5, 0.5])


class EvidenceTest(unittest.TestCase):
    def test_race_evidence_flags_the_slow_race(self):
        ev = race_evidence([12600, 12500, 13500, 12550])
        self.assertEqual([e.direction for e in ev], [0, 0, 1, 0])

    def test_scale_floor_keeps_small_differences_typical(self):
        ev = race_evidence([12600, 12610, 12660])
        self.assertTrue(all(e.direction == 0 for e in ev))

    def test_check_change_verdicts(self):
        slow, typical, fast = race_evidence([12600, 12500, 13500, 12550, 11800])[2], \
            race_evidence([12600, 12500, 13500, 12550])[0], race_evidence([12600, 12500, 13500, 11000])[3]
        self.assertEqual(check_change(slow, 0.5, 0.2).verdict, "supported")
        self.assertEqual(check_change(typical, 0.5, 0.2).verdict, "your_knowledge")
        self.assertEqual(check_change(fast, 0.5, 0.2).verdict, "against_data")
        self.assertEqual(check_change(slow, 0.5, 0.8).verdict, "against_data")
        self.assertEqual(check_change(slow, 0.5, 0.5).verdict, "no_change")

    def test_influence_sums_to_zero_and_leave_out_is_exact(self):
        t, w = [12000.0, 12600.0, None, 12300.0], [0.5, 1.0, 0.5, 0.2]
        inf = influence(t, w)
        self.assertAlmostEqual(sum(x.pull_s for x in inf), 0.0, places=6)
        f = (0.5 * 12000 + 1.0 * 12600 + 0.2 * 12300) / 1.7
        without0 = (1.0 * 12600 + 0.2 * 12300) / 1.2 - f
        self.assertAlmostEqual(inf[0].without_s, without0, places=6)
        self.assertIsNone(inf[2].without_s)

    def test_steering_index(self):
        self.assertIsNone(steering_index(12000, 12300, None).index)
        self.assertAlmostEqual(steering_index(12000, 12300, 12600).index, 0.5)
        self.assertIn("matches your own estimate", steering_index(12000, 12600, 12600).label)
        self.assertIn("beyond", steering_index(12000, 12900, 12600).label)


class SessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coach = CoachService(helpers.model(), helpers.runners(), Repository(":memory:"))
        cls.rid = next(r.runner_id for r in helpers.runners() if len(r.recent(5)) >= 4)

    def session(self, **kw):
        return SteeringSession(PlanSession(self.coach, self.rid), **kw)

    def year(self, s, i):
        return s.races[i]["year"]

    def test_hear_moves_the_slider_and_logs_the_reason(self):
        s = self.session()
        y = self.year(s, 0)
        t = s.hear(f"{y} was a hot day")
        self.assertEqual(s.plan.weights[0], 0.2)
        self.assertEqual(len(t.applied), 1)
        self.assertIn('You said: "hot day"', s.plan.changes[0].text())
        self.assertIn(t.applied[0].verdict, ("supported", "your_knowledge", "against_data"))
        self.assertEqual(s.reasoned_share(), 1.0)

    def test_undo_restores_all_weights_of_the_last_turn(self):
        s = self.session()
        s.hear("trust the recent races more")
        before = list(s.plan.weights)
        s.hear(f"ignore {self.year(s, 0)}")
        s.hear("scratch that")
        self.assertEqual(s.plan.weights, before)

    def test_recency_keeps_a_race_the_coach_lowered(self):
        s = self.session()
        oldest = min(range(len(s.races)), key=lambda i: s.races[i]["year"])
        s.hear(f"ignore {s.races[oldest]['year']}")
        s.hear("trust the recent races more")
        self.assertEqual(s.plan.weights[oldest], 0.0)

    def test_at_least_one_race_must_count(self):
        s = self.session()
        for i in range(len(s.races)):
            s.hear(f"set {s.races[i]['year']} to 0")
        self.assertTrue(any(w > 0 for w in s.plan.weights))
        self.assertTrue(any("left at" in n for t in s.turns for n in t.notes))

    def test_manual_moves_count_as_unreasoned(self):
        s = self.session()
        s.hear(f"{self.year(s, 0)} was a hot day")
        s.manual(1, 0.9)
        self.assertAlmostEqual(s.reasoned_share(), 0.5)

    def test_estimate_feeds_the_steering_index(self):
        s = self.session()
        s.hear("I think she can run 2:59")
        st = s.state()
        self.assertEqual(st["steering"]["coach_estimate_s"], 2 * 3600 + 59 * 60)
        self.assertIsNotNone(st["steering"]["label"])
        json.dumps(st)                                  # the state must be JSON-ready for the API

    def test_replay_gives_identical_weights(self):
        s = self.session()
        for text in ["trust the recent races more", f"{self.year(s, 1)} was terrible", "scratch that",
                     f"set {self.year(s, 0)} to 0.7"]:
            s.hear(text)
        again = replay(PlanSession(self.coach, self.rid), s.transcript())
        self.assertEqual(again.plan.weights, s.plan.weights)
        self.assertAlmostEqual(again.plan.model_finish, s.plan.model_finish)


class BenchmarkTest(unittest.TestCase):
    def test_dev_split_is_fully_correct(self):
        self.assertEqual(run_benchmark().accuracy("dev"), 1.0)

    def test_test_split_floor_and_no_wrong_races(self):
        r = run_benchmark()
        precision, _ = r.race_scores("test")
        self.assertGreaterEqual(precision, 0.95)          # the rules may miss things, but must not move wrong races
        self.assertGreaterEqual(r.accuracy("test"), 0.5)  # regression floor; the honest number is reported, not tuned


class CheckedAITest(unittest.TestCase):
    def test_invented_quotes_and_races_are_dropped(self):
        data = {"changes": [{"race": 2, "direction": "down", "strength": "strong", "quote": "broken toe"},
                            {"race": 9, "direction": "down", "quote": "broken toe"},
                            {"race": 1, "direction": "down", "quote": "she was sick"},
                            {"race": 3, "direction": "sideways", "quote": "2016"}],
                "command": None, "estimate": "3:20:00"}
        it = validate_proposal(data, "She ran 2016 with a broken toe", RACES, None)
        self.assertEqual([(a.races, a.value) for a in it.actions], [((2,), -0.3)])
        self.assertEqual(len(it.unresolved), 3)
        self.assertEqual(it.estimate_s, 3 * 3600 + 20 * 60)

    def test_hybrid_uses_the_ai_and_falls_back_to_rules(self):
        from pacesense.llm.service import AIService
        fake = FakeOpenRouter()
        ai = AIService(Repository(":memory:"), env={"OPENROUTER_API_KEY": "sk-or-test"}, client=fake.client())
        fake.chats.append(chat_reply(json.dumps({"changes": [{"race": 2, "direction": "down", "strength": "strong",
                                                              "quote": "broken toe"}], "command": None})))
        it, used = interpret_hybrid(ai, "she ran 2016 with a broken toe", RACES)
        self.assertEqual(used, "ai")
        self.assertEqual(it.actions[0].races, (2,))
        self.assertIn("broken toe", fake.requests[-1]["body"]["messages"][-1]["content"])
        fake.chats.append((500, "server error"))
        it, used = interpret_hybrid(ai, "ignore 2016", RACES)
        self.assertTrue(used.startswith("rules (AI failed"))
        self.assertEqual(it.actions[0].value, 0.0)
        offline = AIService(Repository(":memory:"), env={"PACESENSE_AI": "offline"})
        self.assertEqual(interpret_hybrid(offline, "ignore 2016", RACES)[1], "rules")


class RecencyFitTest(unittest.TestCase):
    def test_fit_is_reproducible(self):
        from pacesense.steer.recency import fit_half_life
        a = fit_half_life(helpers.model(), helpers.runners(), n=60, seed=5)
        b = fit_half_life(helpers.model(), helpers.runners(), n=60, seed=5)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertGreater(a.n, 20)
        self.assertIn(a.best_half_life, a.errors)


if __name__ == "__main__":
    unittest.main()
