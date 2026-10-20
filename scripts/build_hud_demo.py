"""Build the command center as one self-contained page that runs in any browser, with no server.

    python scripts/build_hud_demo.py                  # writes docs/command-center-demo.html
    python scripts/build_hud_demo.py --fragment out.html   # the same, without <html>/<head>/<body> (for hosts that add them)

It embeds the demo runner (the same hidden-best-race demo as `pacesense hud`), the interpreter's lexicon, the
browser port of the interpreter and the in-browser session engine (resources/hud_local.js). The numbers it shows
are checked against the Python session by tests/test_hud_demo.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pacesense import racewatch  # noqa: E402
from pacesense.cbr import CaseBase, MarathonModel  # noqa: E402
from pacesense.coach import CoachService  # noqa: E402
from pacesense.data import load_csv  # noqa: E402
from pacesense.hud import HudSession  # noqa: E402
from pacesense.storage import Repository  # noqa: E402
from pacesense.tuning import configured_cbr  # noqa: E402
from pacesense.web.steer_api import demo_runner_id  # noqa: E402

RES = ROOT / "src" / "pacesense" / "resources"


def demo_session(data: Path = ROOT / "data" / "races.csv") -> HudSession:
    runners, _ = load_csv(data)
    coach = CoachService(MarathonModel(CaseBase(runners, configured_cbr())), runners, Repository(":memory:"))
    return HudSession(coach, demo_runner_id(coach), holdout=True)


def demo_data(s: HudSession) -> dict:
    r = s.coach.runner(s.runner_id)
    st = s.state()
    races = []
    for i, (x, p) in enumerate(zip(r.recent(5), s._per_race)):
        races.append({"event": x.event, "year": x.year, "finish_s": round(x.finish_s, 1), "finish": st["races"][i]["finish"],
                      "pred_s": p.finish_s, "cases": [float(v) for v in p.case_finish_s],
                      "cw": [float(v) for v in p.case_weights], "n": int(p.n_neighbours),
                      "profile": None if p.profile is None else [float(v) for v in p.profile]})
    return {"runner": {"name": f"Demo runner ({r.gender}, {r.age})", "gender": r.gender, "age": r.age},
            "races": races, "truth_s": s.truth_s, "truth_label": s.truth_label,
            "noise": [float(v) for v in np.random.default_rng(7).normal(0, 0.008, 43)],
            "on_plan_s": racewatch.ON_PLAN_S, "slipping": racewatch.SLIPPING_PCT}


def build(fragment: bool = False) -> str:
    page = (RES / "hud.html").read_text(encoding="utf-8")
    lex = yaml.safe_load((RES / "steer_lexicon.yaml").read_text(encoding="utf-8"))
    data = demo_data(demo_session())
    inject = ("<script>\nconst LEX = " + json.dumps(lex, separators=(",", ":")) + ";\nconst DEMO = " +
              json.dumps(data, separators=(",", ":")) + ";\n</script>\n<script>\n" +
              (RES / "steer_interpreter.js").read_text(encoding="utf-8") + "\n</script>\n<script>\n" +
              (RES / "hud_local.js").read_text(encoding="utf-8") + "\n</script>\n")
    i = page.rindex("<script>")
    page = page[:i] + inject + page[i:]
    if fragment:
        page = re.sub(r"<!doctype html>\s*<html[^>]*>\s*<head>\s*", "", page, flags=re.I)
        page = re.sub(r'<meta charset="utf-8">\s*<meta name="viewport"[^>]*>\s*', "", page)
        page = page.replace("</head>\n<body>", "").replace("</body>\n</html>", "")
    return page


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "command-center-demo.html"))
    ap.add_argument("--fragment", help="also write a version without <html>/<head>/<body> here")
    a = ap.parse_args()
    Path(a.out).write_text(build(), encoding="utf-8")
    print(f"Wrote {a.out}")
    if a.fragment:
        Path(a.fragment).write_text(build(fragment=True), encoding="utf-8")
        print(f"Wrote {a.fragment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
