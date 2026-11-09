"""The offline helper: answers common questions with no AI, no API key and no network.

It has the same shape as an AI chat model (it reads the conversation and asks for PaceSense functions), but
it decides with simple word rules and writes answers from templates. It is useful in class, without
internet, and for tests. It understands far less than a real AI: one question at a time, in plain words.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .chat import ModelReply, ToolCall

RUNNER_ID = re.compile(r"\b(R\d{3,}|C-[0-9a-f]{10})\b", re.I)
CLOCK = r"\d{1,2}:\d{2}(?::\d{2})?"
TOPICS = [("range", "range"), ("likely", "range"), ("equal", "equal"), ("weight", "weights"), ("similar", "cases"),
          ("case", "cases"), ("target", "target"), ("strateg", "strategies"), ("interval", "intervals"),
          ("pac", "pacing"), ("verdict", "verdict"), ("realistic", "verdict"), ("ambitious", "verdict"),
          ("cautious", "verdict"), ("history", "history"), ("change", "changes"), ("steer", "changes"),
          ("watch", "watch"), ("predict", "prediction"), ("model", "prediction")]
HELP_TEXT = ("I am the offline helper, so I understand a few kinds of question:\n"
             "- \"Predict R00042\" or \"What is the likely range for this runner?\"\n"
             "- \"Is my plan realistic?\" (about the plan open on Plan a race)\n"
             "- \"At 21.1 km in 1:41:00, am I on plan?\"\n"
             "- \"Explain the race weights\", \"Show my saved plans\", \"Find runner Anna\"\n"
             "- \"Add runner Anna de Vries, female, 34: Amsterdam 2023 3:35:48; Rotterdam 2024 3h33\"\n\n"
             "With a BOSON_API_KEY configured on the server, PaceSense can answer free questions and longer conversations.")


class OfflineModel:
    label = "Offline helper (no AI)"

    def __init__(self) -> None:
        self._n = 0

    def _call(self, *calls: tuple[str, dict[str, Any]]) -> ModelReply:
        out = []
        for tool, args in calls:
            self._n += 1
            out.append(ToolCall(f"offline-{self._n}", tool, args))
        return ModelReply("", out)

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        turn = messages[last_user + 1:]
        if any(m["role"] == "tool" for m in turn):
            calls = {c["id"]: c["function"] for m in turn if m["role"] == "assistant" for c in m["tool_calls"]}
            done = [(calls[m["tool_call_id"]]["name"], json.loads(calls[m["tool_call_id"]]["arguments"]),
                     json.loads(m["content"])) for m in turn if m["role"] == "tool"]
            name, args, result = done[-1]
            if name == "check_runner_details" and result.get("ok"):
                return self._call(("propose_runner", args))     # checked and fine: prepare it for the coach
            if any(n == "propose_runner" for n, _, _ in done):     # the check is repeated in the proposal
                done = [d for d in done if d[0] != "check_runner_details"]
            return ModelReply("\n\n".join(_answer(n, r) for n, _, r in done))
        prior = "\n".join(m.get("content", "") for i, m in enumerate(messages)
                            if m.get("role") == "user" and i != last_user)
        return self._decide(messages[last_user]["content"], messages[0]["content"], prior)

    def _decide(self, text: str, system: str, history: str = "") -> ModelReply:
        t = text.lower()
        conversation = (history + "\n" + text).lower()
        has_non_marathon_event = bool(re.search(
            r"\b(trial run|training run|training session|relay|time trial|track session|parkrun)\b", conversation))
        has_injury = bool(re.search(r"\b(injur\w*|pain|hurt|sore|strain|sprain|fracture)\b", conversation))
        asks_about_injury_impact = bool(
            re.search(r"\b(injur\w*|pain|hurt|sore|strain|sprain|fracture)\b", t)
            or (has_injury and re.search(r"\b(performance|affect|impact|recovery|recover|training)\b", t))
        )
        mentions_non_marathon_event = bool(re.search(
            r"\b(trial run|training run|training session|relay|time trial|track session|parkrun)\b", t))
        if has_non_marathon_event and (asks_about_injury_impact or mentions_non_marathon_event):
            if asks_about_injury_impact:
                return ModelReply(
                    "A trial run or relay is not one of her marathon records, so I won't change those race weights. "
                    "A leg injury could affect her marathon performance depending on its severity, when it happened, "
                    "and how fully she has recovered. The current marathon estimate does not measure that injury. "
                    "Is she fully recovered and training normally now?"
                )
            return ModelReply(
                "Got it — that was a trial or training event, not a marathon record in this database. "
                "I'll keep it separate and won't change the marathon weights. What would you like to know about it?"
            )
        rid_m = RUNNER_ID.search(text)
        # Studio identifies the selected runner as "(ID R00004)"; older plan context
        # used "runner_id R00004)". Accept both for contextual questions.
        open_m = re.search(r"\b(?:runner_id\s+|ID\s+)(R\d{3,}|C-[0-9a-f]{10})\b", system, re.I)
        rid = rid_m.group(1).upper() if rid_m else (open_m.group(1).upper() if open_m else None)
        if rid and rid.startswith("C-"):
            rid = "C-" + rid[2:].lower()

        if re.search(r"\b(add|new)\b.*\brunner\b", t):
            return self._call(("check_runner_details", parse_runner_text(text)))

        # Speech recognition can turn a race name into a different word. If the year
        # matches one existing record, ask before treating the unknown name as a race.
        year_m = re.search(r"\b((?:19|20)\d{2})\b", text)
        records_m = re.search(r"Existing marathon records:\s*([^\n]+)", system, re.I)
        if year_m and records_m:
            year = year_m.group(1)
            candidates = [part.strip() for part in records_m.group(1).split(";")
                          if re.search(rf"\b{year}\b", part)]
            mentioned = re.sub(rf"\b{year}\b", "", text, count=1, flags=re.I).strip(" ,.:;?-")
            mentioned = re.sub(r"\b(?:the|race|marathon|in|at|was|were)\b", " ", mentioned, flags=re.I)
            mentioned = " ".join(mentioned.split())
            if (len(mentioned) >= 3
                    and not re.search(r"\b(hot|heat|humid|injur\w*|sick|wind\w*|rain\w*|cold|weather|training|form)\b",
                                      mentioned, re.I)
                    and candidates and not any(
                    re.search(re.escape(re.sub(rf"\s+{year}\b", "", candidate, flags=re.I)), text, re.I)
                    for candidate in candidates)):
                if len(candidates) == 1:
                    return ModelReply(f"I don't see “{mentioned} {year}” in her marathon records. "
                                      f"The record for {year} is {candidates[0]}. Did you mean that race?")
                return ModelReply(f"I don't see “{mentioned} {year}” as an exact race name. "
                                  f"Which of these {year} records did you mean: {', '.join(candidates)}?")
        watch = re.search(r"(\d+(?:\.\d+)?)\s*km\b.*?(" + CLOCK + ")", t) or \
            re.search(r"(" + CLOCK + r").*?(\d+(?:\.\d+)?)\s*km\b", t)
        if watch and re.search(r"watch|ahead|behind|on plan|on pace|at ", t):
            a, b = watch.groups()
            km, clock = (a, b) if ":" not in a else (b, a)
            return self._call(("watch_moment", {"distance_km": float(km), "elapsed": clock}))
        if re.search(r"saved|track record|results?\b", t):
            return self._call(("saved_plans", {}))
        m = re.search(r"\b(?:find|search|look up)\b(?:\s+(?:for|runner|runners))*\s+(.+)", text, re.I)
        if m:
            return self._call(("find_runners", {"query": m.group(1).strip(" ?.")}))
        if re.search(r"\b(my|this|the|current) plan\b", t) and not explain_words(t):
            return self._call(("current_plan", {}))
        explain = re.search(r"\b(explain|what is|what's|what are|meaning|mean|how does|why)\b", t)
        if rid_m:
            if re.search(r"race|history|detail|past|split", t) and not re.search(r"predict|range|target", t):
                return self._call(("runner_details", {"runner_id": rid}))
            return self._call(("predict", {"runner_id": rid}))
        if explain and not re.search(r"\bmy plan\b|\bthis plan\b", t):
            for word, topic in TOPICS:
                if word in t:
                    if topic in ("range", "prediction", "target", "cases") and rid:
                        # The prediction tool returns the active runner's actual range and its meaning.
                        # A second generic explain call would hide that first result in the offline loop.
                        return self._call(("predict", {"runner_id": rid}))
                    return self._call(("explain", {"topic": topic}))
        if re.search(r"my plan|this plan|the plan|current plan|realistic|ambitious|cautious|check", t):
            return self._call(("current_plan", {}))
        if rid and re.search(r"predict|finish|target|range|time|similar", t):
            return self._call(("predict", {"runner_id": rid}))
        return ModelReply(HELP_TEXT)


def explain_words(t: str) -> bool:
    return bool(re.search(r"\b(explain|what is|what's|what are|meaning|mean|how does|why)\b", t))


def parse_runner_text(text: str) -> dict[str, Any]:
    """Best-effort reading of 'Add runner NAME, female, 34: Amsterdam 2023 3:35:48; Rotterdam 2024 3h33'."""
    body = re.split(r"\brunner\b[:\s]*", text, maxsplit=1, flags=re.I)[-1]
    races = []
    race_re = re.compile(r"([A-Za-z][A-Za-z .'\-]*?)\s+((?:19|20)\d{2})\s*(?:in\s+|:\s*|-\s*)?"
                         r"(\d+(?::\d{2}){1,2}|\d+\s*h(?:\s*\d+\s*m?)?(?:\s*\d+\s*s?)?)", re.I)
    for m in race_re.finditer(body):
        event = re.sub(r"^(?:and|ran|then)\s+", "", m.group(1).strip(" ,;:"), flags=re.I)
        races.append({"event": event.strip() or "Marathon", "year": int(m.group(2)), "finish": m.group(3).strip()})
    head = race_re.split(body, maxsplit=1)[0] if races else body
    head = re.split(r"[:;]", body, maxsplit=1)[0] if ":" in body or ";" in body else head
    parts = [p.strip() for p in head.split(",") if p.strip()]
    name = parts[0] if parts else ""
    gender = next((w for w in re.findall(r"[a-z]+", head.lower())
                   if w in ("female", "male", "woman", "man", "f", "m")), None)
    age_m = re.search(r"(?:age[d]?\s*)?\b(\d{2})\b(?:\s*(?:years|yrs|y/o))?", head[len(name):])
    out: dict[str, Any] = {"name": name, "races": races}
    if gender:
        out["gender"] = gender
    if age_m:
        out["age"] = int(age_m.group(1))
    return out


def _answer(name: str, r: dict) -> str:
    if "error" in r:
        return f"PaceSense could not do that: {r['error']}."
    if name == "predict":
        lines = [f"The model's target is **{r['finish']}** (likely range {r['likely_range']}, from "
                 f"{r['similar_runners']} similar runners)."]
        if r.get("range_meaning"):
            lines.append(r["range_meaning"])
        if r["with_equal_weights"] != r["finish"]:
            lines.append(f"With equal race weights it would be {r['with_equal_weights']}.")
        lines += [f"- {x}" for x in r["insights"]]
        return "\n".join(lines)
    if name == "runner_details":
        lines = [f"**{r['name']}**, recent marathons (oldest first):"]
        lines += [f"- {x['race']}: {x['finish']}" for x in r["recent_races_oldest_first"]]
        return "\n".join(lines)
    if name == "current_plan":
        if r.get("open_plan", True) is None:
            return r["note"]
        lines = [f"Your plan for **{r['runner']}** finishes in **{r['plan_finish']}**. The check says "
                 f"**{r['check']}**: {r['check_meaning']}",
                 f"The model's target is {r['model_finish']} (likely range {r['likely_range']})."]
        lines += [f"- Warning: {w}" for w in r["warnings"]]
        if r["changes_newest_first"]:
            lines.append(f"Your last change: {r['changes_newest_first'][0]}")
        return "\n".join(lines)
    if name == "watch_moment":
        return (f"At {r['distance_km']:g} km in {r['elapsed']}: **{r['ahead_or_behind']}** (the plan is at "
                f"{r['plan_elapsed_here']} here). The watch says: \"{r['watch_cue']}\"")
    if name == "saved_plans":
        if not r["plans"]:
            return "There are no saved plans yet."
        lines = [f"{len(r['plans'])} saved plan(s):"]
        lines += [f"- {p['runner']}, {p['race']}: plan {p['plan_finish']}" + (f", result {p['result']}" if p["result"] else "")
                  for p in r["plans"][:8]]
        tr = r["track_record"]
        if tr.get("n"):
            lines.append(f"Across {tr['n']} result(s), your plans were off by {tr['plan_error_pct']:.1f}% on average, "
                         f"the model on its own by {tr['model_default_error_pct']:.1f}%.")
        return "\n".join(lines)
    if name == "find_runners":
        if not r["runners"]:
            return "No runners match. Try another name or runner number."
        return "\n".join(["Runners found:"] + [f"- {x['name']} ({x['runner_id']}), {x['n_races']} marathons, best "
                                                f"{x['best']}" for x in r["runners"]])
    if name == "check_runner_details":
        lines = ["I checked the details."]
        lines += [f"- Corrected: {c}" for c in r["corrections"]]
        lines += [f"- Problem: {p}" for p in r["problems"]]
        lines += [f"- Missing: {m}" for m in r["missing"]]
        lines.append("Send the runner again with these fixed.")
        return "\n".join(lines)
    if name == "propose_runner":
        if not r.get("proposed"):
            return "The runner is not ready yet: " + "; ".join(r.get("problems", []) + r.get("missing", []))
        lines = ["The runner is ready. Check the details below and press **Add this runner**."]
        lines += [f"- Corrected: {c}" for c in r["corrections"]]
        return "\n".join(lines)
    if name == "explain":
        return f"**{r['title']}.** {r['explanation']}\n\n_{r['source']}_"
    return json.dumps(r)
