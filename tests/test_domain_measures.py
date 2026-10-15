import unittest

import numpy as np

from pacesense.domain import (MARATHON_KM, SEGMENT_KM, Race, Runner, ValidationError, format_time,
                              linear_pacing_profile, parse_time)
from pacesense.measures import acceptance, adaptation_class, compute_trial_measures, model_adaptation, percent_error

T = parse_time


class TimeTests(unittest.TestCase):
    def test_parse_and_format(self):
        self.assertEqual(parse_time("3:31:20"), 12680)
        self.assertEqual(parse_time("3:31"), 12660)
        self.assertEqual(parse_time("12680"), 12680)
        self.assertIsNone(parse_time("abc"))
        self.assertIsNone(parse_time(""))
        self.assertEqual(format_time(12680), "3:31:20")
        self.assertEqual(format_time(12680, with_seconds=False), "3:31")

    def test_race_validation(self):
        with self.assertRaises(ValidationError):
            Race("X", 2019, 3000)
        with self.assertRaises(ValidationError):
            Race("X", 2019, 12000, segments_s=(1.0,) * 8)

    def test_profile_is_relative_pace(self):
        splits = [1500 * i for i in range(1, 9)]
        race = Race.from_cumulative_splits("A", 2019, 12660, splits)
        prof = race.profile()
        self.assertAlmostEqual(float(np.dot(prof, SEGMENT_KM) / MARATHON_KM), 1.0, places=9)
        self.assertEqual(len(prof), 9)

    def test_runner_sorts_races(self):
        r = Runner("x", "F", [Race("B", 2018, 13000), Race("A", 2015, 14000)])
        self.assertEqual([x.year for x in r.races], [2015, 2018])

    def test_pacing_profile(self):
        even = linear_pacing_profile(0)
        self.assertTrue(np.allclose(even, 1))
        fast = linear_pacing_profile(1)
        self.assertGreater(fast[-1], fast[0])  # start fast -> slower at the end
        with self.assertRaises(ValidationError):
            linear_pacing_profile(2)


class MeasureTests(unittest.TestCase):
    """Table 3 of the paper (times in h:mm)."""

    def test_table3_cases(self):
        # Coach 24: 4:00, 3:41, 3:52, 3:50 -> acceptance 1 (5-minute rounding), adaptation 0.57
        self.assertEqual(acceptance(T("3:52"), T("4:00"), T("3:50")), 1.0)
        self.assertAlmostEqual(model_adaptation(T("3:41"), T("3:52"), T("4:00")), 11 / 19, places=6)
        # Coach 3: steered the model fully to their own estimate
        self.assertEqual(model_adaptation(T("3:29"), T("3:15"), T("3:15")), 1.0)
        self.assertEqual(adaptation_class(1.0, True), 3)
        # Coach 47: over-adaptation
        a = model_adaptation(T("3:28"), T("3:20"), T("3:25"))
        self.assertGreater(a, 1.25)
        self.assertEqual(adaptation_class(a, True), 4)
        self.assertEqual(adaptation_class(None, False), 0)

    def test_acceptance_bounds(self):
        self.assertEqual(acceptance(T("3:30"), T("3:50"), T("3:50")), 0.0)          # stayed put
        self.assertEqual(acceptance(T("3:30"), T("3:50"), T("4:10")), 0.0)          # moved away -> capped
        self.assertAlmostEqual(acceptance(T("3:30"), T("3:50"), T("3:41")), 0.45)   # partial (no rounding match)
        self.assertIsNone(acceptance(T("3:30"), T("3:30"), T("3:42")))              # undefined

    def test_trial_measures(self):
        m = compute_trial_measures(interactive=False, coach_pre_s=T("3:40"), model_initial_s=T("3:30"),
                                   model_adapted_s=None, coach_post_s=T("3:30"), truth_s=T("3:28"))
        self.assertEqual(m.acceptance, 1.0)
        self.assertEqual(m.adaptation_class, 0)
        self.assertIsNone(m.err_model_adapted)
        self.assertAlmostEqual(m.err_model_initial, percent_error(T("3:28"), T("3:30")))


if __name__ == "__main__":
    unittest.main()
