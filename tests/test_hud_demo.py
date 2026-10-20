"""The self-contained command-center page (scripts/build_hud_demo.py) builds and carries what it needs.

Its numbers are checked against the Python session in a browser when the page is rebuilt (the build is
deterministic; the JavaScript engine is a port of hud/session.py and was matched to it to 4 decimals)."""
import importlib.util
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location("build_hud_demo", ROOT / "scripts" / "build_hud_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DemoPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = load_builder()
        cls.page = cls.b.build()

    def test_page_embeds_engine_lexicon_and_runner(self):
        for needle in ("window.PACESENSE_LOCAL", "const Steer = (() =>", "const LEX = {", "const DEMO = {",
                       "if (window.PACESENSE_LOCAL) return window.PACESENSE_LOCAL"):
            self.assertIn(needle, self.page)
        demo = json.loads(re.search(r"const DEMO = (\{.*?\});\n", self.page).group(1))
        self.assertGreaterEqual(len(demo["races"]), 2)
        self.assertNotIn(round(demo["truth_s"]), [round(r["finish_s"]) for r in demo["races"]])   # best race hidden
        self.assertEqual(len(demo["noise"]), 43)

    def test_build_is_deterministic(self):
        self.assertEqual(self.page, self.b.build())

    def test_fragment_has_no_document_wrapper(self):
        frag = self.b.build(fragment=True)
        self.assertTrue(frag.lstrip().startswith("<title>"))
        for tag in ("<!doctype", "<html", "<body>", "</html>"):
            self.assertNotIn(tag, frag.lower())


if __name__ == "__main__":
    unittest.main()
