import unittest

from matplotlib.figure import Figure

from pacesense import racewatch as rw
from pacesense.dashboard import charts as c

PACES = [300, 302, 304, 305, 306, 308, 310, 312, 314]


class ChartTests(unittest.TestCase):
    def test_pace_chart_faster_is_up_and_labelled(self):
        fig = c.pace_chart(PACES, [p + 3 for p in PACES], [("Rotterdam 2014", [p + 20 for p in PACES])])
        ax = fig.axes[0]
        lo, hi = ax.get_ylim()
        self.assertGreater(lo, hi)                                   # inverted axis
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(labels, ["Rotterdam 2014", "Model plan", "Your plan"])
        self.assertIn("min per km", ax.get_ylabel())
        self.assertIn("min per mi", c.pace_chart(PACES, PACES, unit="mi").axes[0].get_ylabel())
        self.assertTrue(c.to_png(fig).startswith(b"\x89PNG"))

    def test_other_charts_build(self):
        for fig in (c.race_bars(PACES, title="Race"), c.sparkline(PACES), c.sparkline(None),
                    c.range_chart(11000, 11800, 11400, 11300), c.range_chart(11000, 11800, 11400, 11400)):
            self.assertIsInstance(fig, Figure)
            self.assertTrue(c.to_png(fig, dpi=50).startswith(b"\x89PNG"))

    def test_watch_face_texts(self):
        def texts(st):
            return " | ".join(t.get_text() for t in c.watch_face(st, runner="Ann").axes[0].texts)
        t10 = rw.target_elapsed(PACES, 10)
        self.assertIn("ahead", texts(rw.status(PACES, 10, t10 - 60)))
        self.assertIn("behind", texts(rw.status(PACES, 10, t10 + 60)))
        self.assertIn("on plan", texts(rw.status(PACES, 10, t10 + 3)))
        self.assertIn("11 km target", texts(rw.status(PACES, 10, t10)))
        done = texts(rw.status(PACES, 42.195, rw.target_elapsed(PACES, 42.195) - 20))
        self.assertIn("under plan", done)
        self.assertIn("finish time", done)

    def test_ahead_behind_chart(self):
        track = rw.simulate(PACES, "late_fade")
        ax = c.ahead_behind_chart(track, PACES).axes[0]
        lo, hi = ax.get_ylim()
        self.assertEqual(lo, -hi)
        self.assertIsInstance(c.ahead_behind_chart(track, PACES, upto_km=0), Figure)   # nothing run yet


if __name__ == "__main__":
    unittest.main()
