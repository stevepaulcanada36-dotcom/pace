"""Importing Kaggle-style results files and tuning the model on them."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pacesense.data import load_csv, synthetic_runners
from pacesense.domain import format_time
from pacesense.importers import ColumnMap, import_files, normalise_name, write_import
from pacesense.tuning import tune, write_model_config

BOSTON_COLS = ["Bib", "Name", "Age", "M/F", "City", "State", "Country", "5K", "10K", "15K", "20K", "Half", "25K",
               "30K", "35K", "40K", "Pace", "Proj Time", "Official Time", "Overall", "Gender", "Division"]


def boston_files(folder: Path, n: int = 400) -> list[Path]:
    """Synthetic runners written as Kaggle's Boston 2015-2017 files: one row per finisher per year, no runner id."""
    runners = [r for r in synthetic_runners(n, seed=3) if len(r.races) >= 2]
    files = {y: [] for y in (2015, 2016, 2017)}
    for k, r in enumerate(runners):
        for j, race in enumerate(r.races[-3:]):
            year = 2015 + j + (3 - len(r.races[-3:]))
            first, last = "Ann", "Smith" + "".join(chr(97 + (k // 26 ** i) % 26) for i in range(3))
            row = {"Bib": k, "Name": f"{last}, {first}", "Age": 30 + k % 30 + (year - 2015), "M/F": r.gender,
                   "City": "X", "State": "", "Country": "NED", "Official Time": format_time(race.finish_s),
                   "Half": "", "Pace": "", "Proj Time": "-", "Overall": 1, "Gender": 1, "Division": 1}
            cum = [sum(race.segments_s[:i + 1]) for i in range(8)] if race.segments_s else [None] * 8
            for km, c in zip((5, 10, 15, 20, 25, 30, 35, 40), cum):
                row[f"{km}K"] = format_time(c) if c else "-"
            files[year].append(row)
    # same name, gender and birth year twice in one race: must not be linked
    files[2016].append({**files[2016][0], "Bib": 99999})
    files[2015].append({**files[2015][1], "Official Time": "", "Bib": 99998})          # no finish: dropped
    files[2017].append({**files[2017][2], "Name": "Broken, Zed", "10K": "0:01:00", "Bib": 99997})  # broken splits: kept, no splits
    out = []
    for y, rows in files.items():
        p = folder / f"marathon_results_{y}.csv"
        pd.DataFrame(rows, columns=BOSTON_COLS).to_csv(p, index=False)
        out.append(p)
    return out


class ImportTests(unittest.TestCase):
    def test_names_are_matched_whatever_the_order(self):
        self.assertEqual(normalise_name("Desisa, Lelisa"), normalise_name("Lelisa  DESISA"))
        self.assertEqual(normalise_name("José Pérez-López"), "jose lopez perez")

    def test_columns_are_detected(self):
        m = ColumnMap.detect(BOSTON_COLS)
        self.assertEqual((m.name, m.gender, m.age, m.finish), ("Name", "M/F", "Age", "Official Time"))
        self.assertEqual(m.splits[40], "40K")

    def test_boston_files_import_and_link_runners(self):
        with tempfile.TemporaryDirectory() as d:
            paths = boston_files(Path(d))
            df, rep = import_files([str(p) for p in paths])
            self.assertEqual(set(df["event"]), {"Boston"})
            self.assertEqual(set(df["year"]), {2015, 2016, 2017})
            self.assertEqual(rep.dropped.get("finish missing or outside 2:00-7:00"), 1)
            self.assertGreater(rep.runners_with_2plus, 100)
            self.assertEqual(rep.ambiguous_names, 2)
            # every linked runner has at most one race per year, with ages that fit one birth year
            for _, g in df.groupby("runner_id"):
                self.assertEqual(len(g), g["year"].nunique())
                self.assertLessEqual((g["year"] - g["age"]).max() - (g["year"] - g["age"]).min(), 1)
            out = Path(d) / "boston.csv"
            write_import([str(p) for p in paths], str(out))
            runners, _ = load_csv(out)                       # the result loads like any PaceSense race file
            self.assertGreater(sum(len(r.races) >= 2 for r in runners), 100)
            self.assertTrue(Path(str(out) + ".import.json").is_file())

    def test_missing_finish_column_is_explained(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x_2016.csv"
            pd.DataFrame([{"Name": "a", "Age": 30, "Sex": "F"}]).to_csv(p, index=False)
            with self.assertRaises(ValueError) as e:
                import_files([str(p)])
            self.assertIn("--map finish=", str(e.exception))
            df, _ = import_files([str(p)], column_map={"finish": "Name"})   # override is honoured (then dropped)
            self.assertEqual(len(df), 0)


class TuneTests(unittest.TestCase):
    def test_tune_is_reproducible_and_reports_held_out_error(self):
        runners = synthetic_runners(1500, seed=4)
        grid = {"k": (5, 15), "time_window": (0.05,), "w_profile": (0.5,), "w_age": (0.2,)}
        a = tune(runners, n=60, seed=9, grid=grid)
        b = tune(runners, n=60, seed=9, grid=grid)
        self.assertEqual(a.best, b.best)
        self.assertEqual(a.test_error_pct, b.test_error_pct)
        self.assertEqual(len(a.grid), 2)
        self.assertIn("Held-out test error", a.text())

    def test_write_model_config_keeps_other_settings(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pacesense.yaml"
            p.write_text("# top\nmodels:\n  assistant:\n    model: x\nmodel:\n  k: 3\nvoice:\n  language: null\n")
            write_model_config({"k": 25, "time_window": 0.08}, p)
            t = p.read_text()
            self.assertIn("models:\n  assistant:", t)
            self.assertIn("voice:\n  language: null", t)
            self.assertIn("  k: 25", t)
            self.assertNotIn("  k: 3", t)


if __name__ == "__main__":
    unittest.main()
