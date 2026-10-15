import json
import unittest

from pacesense import racewatch as rw
from pacesense.coach import CoachService
from pacesense.domain import MARATHON_KM, ValidationError
from pacesense.planning import PlanSession, change_text, duration_text, pace_text, parse_pace
from pacesense.storage import Repository

from .helpers import model, runners


def coach():
    return CoachService(model(), runners(), Repository(":memory:"))


def a_runner():
    return next(r for r in runners() if len(r.races) >= 4 and all(x.segments_s for x in r.races))


class FormatTests(unittest.TestCase):
    def test_pace_and_duration_text(self):
        self.assertEqual(pace_text(299.6), "5:00")                 # rounds first, never 4:60
        self.assertEqual(pace_text(300, "mi"), "8:03")             # 300 s/km x 1.609
        self.assertAlmostEqual(parse_pace("8:03", "mi"), 483 / 1.609344)
        self.assertEqual(duration_text(82), "1:22")
        self.assertEqual(duration_text(3723), "1:02:03")
        self.assertEqual(change_text(-82), "1:22 faster")
        self.assertEqual(change_text(0.4), "no change")
        for bad in ("4.59", "4:60", "abc", ""):
            with self.assertRaises(ValidationError):
                parse_pace(bad)


class PlanSessionTests(unittest.TestCase):
    def setUp(self):
        self.s = PlanSession(coach(), a_runner().runner_id)

    def test_starts_from_model_with_equal_weights(self):
        self.assertEqual(self.s.source, "model")
        self.assertTrue(all(w == 0.5 for w in self.s.weights))
        self.assertAlmostEqual(self.s.plan_finish, self.s.model_finish, delta=1)
        self.assertTrue(self.s.total_check())
        self.assertEqual(self.s.changes, [])

    def test_weight_change_is_recorded_and_explained(self):
        c = self.s.set_weight(0, 0.1)
        self.assertIsNotNone(c)
        self.assertIn("changed from 0.5 to 0.1", c.text())
        self.assertEqual(self.s.changes[0], c)
        self.assertNotEqual(round(c.model_delta), 0)
        self.assertEqual(round(self.s.plan_finish), round(self.s.model_finish))   # plan follows the model
        self.assertIsNone(self.s.set_weight(0, 0.1))                             # same value: no change

    def test_repeated_moves_merge_and_noop_disappears(self):
        self.s.set_weight(0, 0.3)
        self.s.set_weight(0, 0.1)
        self.assertEqual(len(self.s.changes), 1)
        self.assertIn("from 0.5 to 0.1", self.s.changes[0].text())
        self.s.set_weight(0, 0.5)                                                # back where it started
        self.assertEqual(self.s.changes, [])

    def test_strategy_and_pace_edits(self):
        c = self.s.choose_strategy("negative")
        self.assertIn("from Model plan to Negative split", c.text())
        first = self.s.paces[0]
        c2 = self.s.set_pace(0, first - 20)
        self.assertEqual(self.s.source, "custom")
        self.assertLess(c2.plan_delta, 0)
        self.assertTrue(self.s.total_check())
        # a weight change now leaves the edited plan alone and says so
        c3 = self.s.set_weight(1, 0.9)
        self.assertIn("stayed as it was", c3.text())
        self.assertEqual(round(c3.plan_delta), 0)

    def test_set_pace_texts_validates_everything_first(self):
        texts = [r["pace"] for r in self.s.interval_rows()]
        texts[2] = "9:99"
        before = list(self.s.paces)
        with self.assertRaises(ValidationError):
            self.s.set_pace_texts(texts)
        self.assertEqual(self.s.paces, before)
        texts[2] = pace_text(before[2] - 10)
        changes = self.s.set_pace_texts(texts)
        self.assertEqual(len(changes), 1)

    def test_units(self):
        self.s.set_unit("mi")
        self.assertTrue(all("/mi" not in r["pace"] for r in self.s.interval_rows()))
        self.assertTrue(any("s/mi" in st["description"] for st in self.s.strategies))
        with self.assertRaises(ValidationError):
            self.s.set_unit("m")

    def test_save_and_reopen(self):
        self.s.set_weight(0, 0.2)
        self.s.set_pace(3, self.s.paces[3] - 5)
        plan = self.s.save("Toronto 2026", "2026-10-18", "note")
        again = PlanSession.from_saved_plan(self.s.coach, plan)
        self.assertEqual(again.weights, self.s.weights)
        self.assertEqual([round(x, 3) for x in again.paces], [round(x, 3) for x in self.s.paces])
        self.assertEqual(again.source, "custom")
        text = self.s.plan_text("Toronto 2026", "2026-10-18", "note")
        self.assertIn("Target finish", text)
        self.assertIn("Notes: note", text)


class RaceWatchTests(unittest.TestCase):
    def setUp(self):
        self.paces = [300, 302, 304, 305, 306, 308, 310, 312, 314]
        self.finish = sum(p * k for p, k in zip(self.paces, [5] * 8 + [2.195]))

    def test_splits_add_up(self):
        sp = rw.km_splits(self.paces)
        self.assertEqual(len(sp), 43)
        self.assertEqual(sp[-1].marker, "Finish")
        self.assertAlmostEqual(sp[-1].elapsed_s, self.finish)
        self.assertAlmostEqual(sum(s.split_s for s in sp), self.finish)
        self.assertEqual(sp[4].pace_s, 300)        # km 5 is the last km of the first interval
        self.assertEqual(sp[5].pace_s, 302)        # km 6 starts the second
        self.assertAlmostEqual(sp[-1].split_s, 0.195 * 314)

    def test_target_elapsed_interpolates(self):
        self.assertEqual(rw.target_elapsed(self.paces, 0), 0)
        self.assertAlmostEqual(rw.target_elapsed(self.paces, 7.5), 5 * 300 + 2.5 * 302)
        self.assertAlmostEqual(rw.target_elapsed(self.paces, MARATHON_KM), self.finish)

    def test_status_levels(self):
        t10 = rw.target_elapsed(self.paces, 10)
        self.assertEqual(rw.status(self.paces, 0, 0).level, "start")
        on = rw.status(self.paces, 10, t10 + 5)
        self.assertEqual(on.level, "on")
        ahead = rw.status(self.paces, 10, t10 - 60)
        self.assertEqual((ahead.level, round(ahead.delta_s)), ("ahead", -60))
        self.assertIn("1:00 ahead", ahead.cue)
        behind = rw.status(self.paces, 10, t10 + 40)
        self.assertEqual(behind.level, "behind")
        self.assertAlmostEqual(behind.projected_finish_s, self.finish + 40)
        self.assertAlmostEqual(behind.required_pace_s, (self.finish - t10 - 40) / (MARATHON_KM - 10))
        late = rw.status(self.paces, 39, rw.target_elapsed(self.paces, 39) + 90)
        self.assertEqual(late.level, "slipping")          # 90 s down with 3 km left needs a much faster pace
        done = rw.status(self.paces, MARATHON_KM, self.finish - 12)
        self.assertEqual(done.level, "done")
        self.assertIn("faster", done.cue)
        self.assertEqual(on.next_marker, "11 km")
        self.assertEqual(rw.status(self.paces, 42.1, 99999).next_marker, "Finish")

    def test_status_validation(self):
        for d, t in ((-1, 0), (43, 0), (5, -3)):
            with self.assertRaises(ValidationError):
                rw.status(self.paces, d, t)
        with self.assertRaises(ValidationError):
            rw.status([300] * 8, 5, 1500)

    def test_simulation_profiles(self):
        on = rw.simulate(self.paces, "on_plan", seed=1)
        fast = rw.simulate(self.paces, "fast_start", seed=1)
        fade = rw.simulate(self.paces, "late_fade", seed=1)
        self.assertEqual(on[-1][0], MARATHON_KM)
        self.assertAlmostEqual(on[-1][1], self.finish, delta=self.finish * 0.02)
        self.assertLess(rw.status(self.paces, 10, dict(fast)[10.0]).delta_s, -30)    # ahead at 10 km
        self.assertGreater(fade[-1][1], on[-1][1] + 60)                              # the fade costs time
        d, t = rw.position_at(on, 1800)
        self.assertAlmostEqual(t, 1800)
        self.assertTrue(5 < d < 7)
        self.assertEqual(rw.position_at(on, 10 ** 6), on[-1])
        with self.assertRaises(ValidationError):
            rw.simulate(self.paces, "sprint")

    def test_exports(self):
        band = rw.pace_band_text(self.paces, "Ann: Toronto", every_km=5)
        self.assertIn("Ann: Toronto", band)
        self.assertIn("Finish", band)
        self.assertEqual(sum(1 for line in band.splitlines() if " km " in line), 8)     # 5, 10, ... 40
        csv_text = rw.splits_csv(self.paces)
        self.assertEqual(len(csv_text.strip().splitlines()), 44)
        doc = json.loads(rw.plan_json(self.paces, "Ann", "Toronto", "2026-10-18"))
        self.assertEqual(doc["format"], "pacesense-plan/1")
        self.assertEqual(len(doc["intervals"]), 9)
        self.assertAlmostEqual(doc["target_finish_s"], round(self.finish, 1))


if __name__ == "__main__":
    unittest.main()
