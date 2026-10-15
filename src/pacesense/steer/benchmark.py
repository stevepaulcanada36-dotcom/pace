"""Score the interpreter on resources/steer_benchmark.yaml.

A case is correct only if *everything* matches: every expected race moves the right way (or to the right value),
no other race moves, and the command and estimate match. Race-level precision and recall are also reported, so a
near miss ("found 2016 but also moved 2014") is visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import yaml

from ..assistant.intake import parse_duration
from .intent import Interpretation, interpret

START = 0.5


@dataclass
class CaseResult:
    split: str
    text: str
    ok: bool
    expected: dict
    got: dict
    problems: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    cases: list[CaseResult]

    def split(self, name: str) -> list[CaseResult]:
        return [c for c in self.cases if c.split == name]

    def accuracy(self, name: str) -> float:
        cs = self.split(name)
        return sum(c.ok for c in cs) / len(cs) if cs else float("nan")

    def race_scores(self, name: str) -> tuple[float, float]:
        tp = fp = fn = 0
        for c in self.split(name):
            exp = {y for y, v in c.expected.items() if v != "same"}
            got = set(c.got)
            tp += len(exp & got); fp += len(got - exp); fn += len(exp - got)
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        return prec, rec

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for s in ("dev", "test"):
            p, r = self.race_scores(s)
            out[s] = {"cases": len(self.split(s)), "correct": sum(c.ok for c in self.split(s)),
                      "accuracy": round(self.accuracy(s), 4), "race_precision": round(p, 4), "race_recall": round(r, 4),
                      "failures": [{"text": c.text, "problems": c.problems} for c in self.split(s) if not c.ok]}
        return out


def _load() -> dict:
    return yaml.safe_load(resources.files("pacesense.resources").joinpath("steer_benchmark.yaml")
                          .read_text(encoding="utf-8"))


def _moves(it: Interpretation, races: list[dict]) -> dict[int, float]:
    """Year -> new weight, starting every race at 0.5."""
    w = {r["year"]: START for r in races}
    for a in it.actions:
        for i in a.races:
            y = races[i]["year"]
            if a.kind == "set":
                w[y] = float(a.value)
            elif a.kind == "shift":
                w[y] = round(min(1, max(0, w[y] + float(a.value))), 2)
    return {y: v for y, v in w.items() if round(v - START, 2) != 0}


def run() -> BenchmarkResult:
    data = _load()
    races = data["races"]
    year_index = {r["year"]: i for i, r in enumerate(races)}
    results = []
    for case in data["cases"]:
        ctx = year_index.get(case.get("context"))
        it = interpret(case["text"], races, ctx)
        got = _moves(it, races)
        exp = {int(k): v for k, v in (case.get("expect") or {}).items()}
        problems = []
        for y, want in exp.items():
            new = got.get(y, START)
            if want == "down" and not new < START:
                problems.append(f"{y} should go down (got {new})")
            elif want == "up" and not new > START:
                problems.append(f"{y} should go up (got {new})")
            elif want == "same" and new != START:
                problems.append(f"{y} should not move (got {new})")
            elif isinstance(want, (int, float)) and abs(new - float(want)) > 1e-9:
                problems.append(f"{y} should be {want} (got {new})")
        for y in got:
            if y not in exp:
                problems.append(f"{y} should not move (got {got[y]})")
        kinds = {a.kind for a in it.actions} - {"set", "shift"}
        want_cmd = case.get("command")
        if want_cmd and want_cmd not in kinds:
            problems.append(f"command '{want_cmd}' not recognised")
        if not want_cmd and kinds:
            problems.append(f"unexpected command {sorted(kinds)}")
        if want_cmd == "recency":
            got = {}
        want_est = case.get("estimate")
        if want_est is not None:
            if it.estimate_s is None or abs(it.estimate_s - parse_duration(want_est)) > 1:
                problems.append(f"estimate should be {want_est} (got {it.estimate_s})")
        elif it.estimate_s is not None:
            problems.append(f"unexpected estimate {it.estimate_s}")
        results.append(CaseResult(case["split"], case["text"], not problems, exp, got, problems))
    return BenchmarkResult(results)


__all__ = ["run", "BenchmarkResult", "CaseResult"]
