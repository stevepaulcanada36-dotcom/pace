"""Test cases and scores for comparing OpenRouter models, task by task (the Model lab).

Each task has a small set of cases with known right answers, checked by code rather than by another model:

    assistant   did it call the right PaceSense function, and does its answer contain the right number?
                Any time (h:mm:ss) in the answer that no function returned counts against it (grounding).
    extract     how many of the expected fields did it fill correctly from a coach's dictation?
    refine      did the refined version keep every number and time, add none, and keep the key words?
    transcribe  word error rate against a reference text (you record the audio in the lab)

A score is 0 to 1; a case passes at 0.8 or more. Speed and cost come from OpenRouter's own report. The
built-in cases are deliberately few and simple; add your own to make the comparison fit your use.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import pandas as pd

from ..assistant.chat import Assistant, AssistantError
from ..assistant.tools import AssistantTools
from ..coach import CoachService
from ..planning import PlanSession
from ..storage import Repository
from .forms import facts
from .openrouter import LLMError, Usage
from .service import AIService

PASS = 0.8
SCORING = {
    "assistant": "Score: 0.4 for calling the right PaceSense function + 0.6 for the right value in the answer; halved "
                 "if the answer states a time no function returned.",
    "extract": "Score: the share of expected fields filled correctly (names, gender, age, each race's year and "
               "finish, dates).",
    "refine": "Score: 0.6 for keeping every number and time without adding any + 0.4 for keeping the key words.",
    "transcribe": "Score: 1 minus the word error rate against what you said.",
}


@dataclass
class Case:
    id: str
    task: str
    input: str
    expected: dict[str, Any]
    form: str = ""                  # for extract and refine
    note: str = ""
    audio: bytes | None = field(default=None, repr=False)
    audio_format: str = "wav"


@dataclass
class CaseResult:
    task: str
    model: str
    case_id: str
    score: float
    latency_s: float
    cost_usd: float | None
    tokens: int
    output: str
    details: list[str]
    error: str | None = None
    at: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return self.error is None and self.score >= PASS


# ---- the built-in cases ---------------------------------------------------------------------------
def _demo_runner(coach: CoachService) -> str:
    own = coach.search("", limit=1)
    if own:
        return own[0]["runner_id"]
    return next(r.runner_id for r in coach.runners if len(r.races) >= 3)


def cases(task: str, coach: CoachService) -> list[Case]:
    if task == "assistant":
        rid = _demo_runner(coach)
        races = coach.overview(rid)["races"]
        n = len(races)
        w = [0.1] + [0.5] * (n - 1)
        oldest = races[0]
        return [
            Case("predict", task, f"What finish time does the model predict for runner {rid}?",
                 {"tool": "predict", "contains": [coach.predict(rid)["finish"]]}),
            Case("weights", task, f"For runner {rid}, set the weight of the oldest race to 0.1 and keep the "
                                  f"others at 0.5. What does the model predict then?",
                 {"tool": "predict", "contains": [coach.predict(rid, w)["finish"]]}),
            Case("cases", task, f"How many similar runners is the prediction for {rid} based on?",
                 {"tool": "predict", "contains": [str(coach.predict(rid)["n_cases"])]}),
            Case("history", task, f"What was {rid}'s finish time in {oldest['event']} {oldest['year']}?",
                 {"tool": "runner_details", "contains": [oldest["finish"]]}),
            Case("plan", task, "Is my plan realistic, ambitious or cautious?",
                 {"tool": "current_plan", "contains_any": ["realistic", "ambitious", "cautious"], "plan": rid}),
            Case("watch", task, "On race day, at 21.1 km the runner's watch shows 1:30:00 elapsed. Is the runner "
                                "ahead of or behind my plan?",
                 {"tool": "watch_moment", "contains_any": ["ahead", "behind"], "plan": rid}),
        ]
    if task == "extract":
        return [
            Case("runner-spoken", task, "Uh so the new runner is Anna de Vries, she's a woman, thirty four years "
                 "old. She ran Amsterdam in 2023 in three hours thirty five minutes forty eight seconds, and "
                 "Rotterdam in 2024 in 3:33:11.", {"name": "Anna de Vries", "gender": "F", "age": 34,
                                                 "races": [["Amsterdam", 2023, "3:35:48"], ["Rotterdam", 2024, "3:33:11"]]},
                 form="runner"),
            Case("runner-short", task, "Add Tom Becker, male, 52. Berlin marathon 2019, 3 hours 58. Berlin 2021, "
                 "4:02:30.", {"name": "Tom Becker", "gender": "M", "age": 52,
                              "races": [["Berlin", 2019, "3:58:00"], ["Berlin", 2021, "4:02:30"]]}, form="runner"),
            Case("runner-no-age", task, "Priya Nair, female. Chicago 2022 in 4:10:05.",
                 {"name": "Priya Nair", "gender": "F", "age": None, "races": [["Chicago", 2022, "4:10:05"]]},
                 form="runner", note="age not said: must stay empty"),
            Case("plan", task, "Target race is the Toronto Waterfront Marathon on October 18th 2026. Notes: she had "
                 "a calf strain in August, so keep the first half conservative.",
                 {"target_race": "Toronto Waterfront", "race_date": "2026-10-18", "notes": "calf"}, form="plan_details"),
            Case("result-words", task, "He finished in three hours twenty one minutes and four seconds.",
                 {"finish": "3:21:04"}, form="result"),
            Case("result-clock", task, "She crossed the line at 3:05:59, really happy with that.",
                 {"finish": "3:05:59"}, form="result"),
        ]
    if task == "refine":
        return [
            Case("question", task, "um so like what do you think, is the plan for R00042 uh realistic or is it too "
                 "fast in the, the first half", {"keep": ["R00042"]}, form="question"),
            Case("runner", task, "anna de vries female 34 she did amsterdam 2023 in 3:35:48 and rotterdam 2024 in "
                 "3:33:11", {"keep": ["Anna", "Amsterdam", "Rotterdam"]}, form="runner"),
            Case("notes", task, "she had a calf strain in august, back running 3 weeks now, long runs up to 32 km, "
                 "wants to go sub 3:30", {"keep": ["calf", "32"]}, form="plan_details"),
            Case("result", task, "finished 3:21:04 hot day walked a bit after 35k",
                 {"keep": ["3:21:04", "35"]}, form="result"),
        ]
    if task == "transcribe":
        return []          # needs audio: record it with a reference text in the Model lab
    raise KeyError(task)


# ---- scoring -----------------------------------------------------------------------------------------
_HMS = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")


def score_assistant(case: Case, answer_text: str, steps: list) -> tuple[float, list[str]]:
    exp, details = case.expected, []
    called = [s.name for s in steps]
    tool_ok = exp["tool"] in called
    details.append(f"called {', '.join(called) or 'nothing'}; expected {exp['tool']}")
    low = answer_text.lower()
    if "contains" in exp:
        # whole values only: "5" must not match inside "15" or "3:51:20"
        hit = all(re.search(rf"(?<![\w:.]){re.escape(x.lower())}(?![\w:])", low) for x in exp["contains"])
        details.append(("has" if hit else "missing") + f" the right value {', '.join(exp['contains'])}")
    else:
        hit = any(x in low for x in exp["contains_any"])
        details.append("answers the question" if hit else f"does not say {' / '.join(exp['contains_any'])}")
    results = json.dumps([s.result for s in steps])
    loose = [t for t in _HMS.findall(answer_text) if t not in results]
    score = 0.4 * tool_ok + 0.6 * hit
    if loose:
        score *= 0.5
        details.append(f"times not returned by any function: {', '.join(loose)}")
    return round(score, 3), details


def _same_text(a: Any, b: Any) -> bool:
    return str(a or "").strip().lower() == str(b or "").strip().lower()


def score_extract(case: Case, fields: dict) -> tuple[float, list[str]]:
    exp, checks, details = case.expected, [], []

    def check(name: str, ok: bool, got: Any) -> None:
        checks.append(ok)
        if not ok:
            details.append(f"{name}: got {got!r}")
    if case.form == "runner":
        check("name", _same_text(fields.get("name"), exp.get("name")), fields.get("name"))
        check("gender", fields.get("gender") == exp.get("gender"), fields.get("gender"))
        check("age", fields.get("age") == exp.get("age"), fields.get("age"))
        got = fields.get("races") or []
        check("number of races", len(got) == len(exp.get("races", [])), len(got))
        for ev, yr, fin in exp.get("races", []):
            m = next((r for r in got if r.get("year") == yr and ev.lower() in str(r.get("event", "")).lower()), None)
            check(f"{ev} {yr}", m is not None and m.get("finish") == fin, m and m.get("finish"))
    elif case.form == "plan_details":
        for name, want in exp.items():
            got = fields.get(name)
            if name == "race_date":
                check(name, got == want, got)
            else:                                        # target race and notes: the expected words appear
                check(name, str(want).lower() in str(got or "").lower(), got)
    elif case.form == "result":
        check("finish", fields.get("finish") == exp.get("finish"), fields.get("finish"))
    score = sum(checks) / len(checks) if checks else 0.0
    details.insert(0, f"{sum(checks)} of {len(checks)} fields right")
    return round(score, 3), details


def score_refine(case: Case, refined: str, added: list[str], dropped: list[str]) -> tuple[float, list[str]]:
    n = max(1, len(facts(case.input)))
    fact_score = max(0.0, 1 - (len(added) + len(dropped)) / n)
    keep = case.expected.get("keep", [])
    kept = [k for k in keep if k.lower() in refined.lower()]
    keep_score = len(kept) / len(keep) if keep else 1.0
    details = [f"kept {len(kept)} of {len(keep)} key words"]
    if added:
        details.append(f"added: {', '.join(added)}")
    if dropped:
        details.append(f"dropped: {', '.join(dropped)}")
    if not refined.strip():
        return 0.0, ["empty answer"]
    if len(refined) > 2.5 * len(case.input) + 40:
        details.append("much longer than the input")
        keep_score *= 0.8
    return round(0.6 * fact_score + 0.4 * keep_score, 3), details


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: (substitutions + deletions + insertions) / words in the reference."""
    norm = lambda s: re.findall(r"[a-z0-9:']+", (s or "").lower())  # noqa: E731
    r, h = norm(reference), norm(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


# ---- running -----------------------------------------------------------------------------------------
def run_case(ai: AIService, task: str, model: str, case: Case, coach: CoachService) -> CaseResult:
    t0 = time.perf_counter()
    usage, output = Usage(), ""
    try:
        if task == "assistant":
            plan = PlanSession(coach, case.expected["plan"]) if case.expected.get("plan") else None
            tools = AssistantTools(coach, plan=lambda: plan)
            ans = Assistant(ai.chat_model(model), tools).ask(case.input)
            usage, output = ans.usage, ans.text
            score, details = score_assistant(case, ans.text, ans.steps)
        elif task == "extract":
            ex = ai.extract(case.form, case.input, model=model)
            usage, output = ex.step.usage, json.dumps(ex.fields)
            score, details = score_extract(case, ex.fields)
        elif task == "refine":
            r = ai.refine(case.form, case.input, model=model)
            usage, output = r.step.usage, r.text
            score, details = score_refine(case, r.text, r.added, r.dropped)
        elif task == "transcribe":
            t = ai.transcribe(case.audio or b"", case.audio_format, model=model)
            usage, output = t.usage, t.text
            e = wer(case.expected["reference"], t.text)
            score, details = round(max(0.0, 1 - e), 3), [f"word error rate {e:.0%} ({t.route})"]
        else:
            raise KeyError(task)
        error = None
    except (LLMError, AssistantError) as e:
        score, details, error = 0.0, [], str(e)
    except Exception as e:  # a malformed answer or case must not stop the whole run
        score, details, error = 0.0, [], f"{type(e).__name__}: {e}"
    return CaseResult(task, model, case.id, score, round(time.perf_counter() - t0, 2), usage.cost,
                      usage.prompt_tokens + usage.completion_tokens, output, details, error)


def run_lab(ai: AIService, task: str, models: list[str], chosen: list[Case], coach: CoachService,
            progress: Callable[[float, str], None] | None = None) -> list[CaseResult]:
    out, total = [], max(1, len(models) * len(chosen))
    for i, model in enumerate(models):
        for j, case in enumerate(chosen):
            if progress:
                progress((i * len(chosen) + j) / total, f"{model} · {case.id}")
            out.append(run_case(ai, task, model, case, coach))
    if progress:
        progress(1.0, "done")
    return out


def summary(results: list[CaseResult]) -> pd.DataFrame:
    """One row per model: average score, cases passed, errors, average seconds, total cost. Best first."""
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame([{**asdict(r), "passed": r.passed} for r in results])
    g = df.groupby("model")
    out = pd.DataFrame({
        "Score": g["score"].mean().round(3),
        "Passed": g["passed"].sum().astype(int).astype(str) + " / " + g.size().astype(str),
        "Errors": g["error"].apply(lambda s: int(s.notna().sum())),
        "Seconds per case": g["latency_s"].mean().round(2),
        "Cost (USD)": g["cost_usd"].apply(lambda s: round(float(s.sum()), 5) if s.notna().any() else None),
    }).sort_values(["Score", "Seconds per case"], ascending=[False, True])
    return out.reset_index().rename(columns={"model": "Model"})


# ---- keeping results ---------------------------------------------------------------------------------
LAB_KEY, LAB_LIMIT = "lab_results", 2000


def save_results(repo: Repository, results: list[CaseResult]) -> None:
    rows = repo.get_setting(LAB_KEY, []) or []
    rows += [{**asdict(r), "passed": r.passed} for r in results]
    repo.set_setting(LAB_KEY, rows[-LAB_LIMIT:])


def saved_results(repo: Repository) -> pd.DataFrame:
    rows = repo.get_setting(LAB_KEY, []) or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["when"] = pd.to_datetime(df["at"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
    df["details"] = df["details"].apply(lambda d: "; ".join(d) if isinstance(d, list) else d)
    return df[["when", "task", "model", "case_id", "score", "passed", "latency_s", "cost_usd", "tokens", "error",
               "details", "output"]]


__all__ = ["Case", "CaseResult", "PASS", "cases", "run_case", "run_lab", "summary", "wer", "save_results",
           "saved_results"]
