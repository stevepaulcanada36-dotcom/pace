"""A command-center session: one runner, the steering session, the assistant, the simulator and the feedback loop.

The feedback loop, in two speeds:

* Immediate. Every change gets a card in plain words: what you said, what moved, whether the runner's own data
  agrees, what it did to the target and the likely range, and two or three next steps (undo, keep, explain).
  You can react to a card (makes sense / not sure); reactions are counted, so the loop also records how often
  the coach agreed with the data checks.
* Outcome. In demo mode the runner's fastest recent race is hidden (the paper's accuracy protocol, Sec. 4.2.2).
  After every change the session knows whether the prediction moved closer to the real result; "reveal" shows
  it. With real runners, saved plans + recorded results feed the coach's track record (coach.track_record()).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..assistant.chat import Assistant, AssistantError
from ..assistant.tools import AssistantTools
from ..cbr import holdout_best
from ..coach import CoachService
from ..domain import SEGMENT_LABELS, ValidationError, format_time
from ..planning import PlanSession
from ..steer import SteeringSession
from .sim import simulate_race

VERDICT_WORDS = {"supported": "the data agrees", "your_knowledge": "this rests on your knowledge",
                 "against_data": "this goes against the data", "undo": "undone", "no_change": "no change"}


def _signed(s: float) -> str:
    r = int(round(abs(s)))
    return f"{'+' if s > 0 else '−'}{r // 60}:{r % 60:02d}"


@dataclass
class HudSession:
    coach: CoachService
    runner_id: str
    holdout: bool = False
    ai: object | None = None            # AIService (offline unless a key is configured)
    ai_reader: bool = False             # let the AI read paraphrases for the sliders (checked by code)
    truth_s: float | None = field(init=False, default=None)
    truth_label: str = field(init=False, default="")
    revealed: bool = field(init=False, default=False)
    reactions: list[dict] = field(init=False, default_factory=list)
    history: list[dict] = field(init=False, default_factory=list)
    assistant_context: str = field(init=False, default="")
    source_runner: str = field(init=False, default="")

    def __post_init__(self) -> None:
        rid = self.runner_id
        self.source_runner = rid
        if self.holdout:
            r = self.coach.runner(rid)
            if len(r.recent(5)) < 3:
                raise ValidationError("demo mode needs a runner with at least 3 races")
            best, shown = holdout_best(r)
            name = f"{r.display_name} (best race hidden)"
            rid = self.coach.add_runner(name, r.gender, r.age, shown)["runner_id"]
            self.truth_s, self.truth_label = best.finish_s, f"{best.event} {best.year}"
        self.plan = PlanSession(self.coach, rid)
        self.runner_id = rid
        self.steer = SteeringSession(self.plan, ai=self.ai if self.ai_reader else None)
        self._per_race = self.coach.model.per_race(self.coach.runner(rid), self.coach.runner(rid).recent(5))

    # ---- assistant ----------------------------------------------------------------------------
    def ask(self, question: str) -> tuple[str, list[dict]]:
        from ..assistant.offline import OfflineModel
        model = self.ai.chat_model() if self.ai is not None else OfflineModel()
        tools = AssistantTools(self.coach, plan=lambda: self.plan)
        try:
            ans = Assistant(model, tools).ask(question, self.history, self.assistant_context)
        except AssistantError:
            # A configured key must not turn a temporary provider outage into an HTTP 500.
            # Fall back to the deterministic local helper for this turn only; never
            # pretend that Boson answered or silently change race weights.
            if self.ai is None or not getattr(self.ai, "online", False):
                raise
            ans = Assistant(OfflineModel(), tools).ask(question, self.history, self.assistant_context)
            ans.text = "Boson is temporarily unavailable, so I used PaceSense's local fallback. " + ans.text
        self.remember_turn(question, ans.text)
        steps = [{"function": s.name, "arguments": s.arguments} for s in ans.steps]
        return ans.text, steps

    def remember_turn(self, question: str, answer: str) -> None:
        """Keep every coach turn, including actions and adaptive follow-up questions, in chat context."""
        if (len(self.history) >= 2 and self.history[-2].get("role") == "user"
                and self.history[-2].get("content") == question):
            self.history[-1] = {"role": "assistant", "content": answer}
        else:
            self.history.extend([{"role": "user", "content": question},
                                 {"role": "assistant", "content": answer}])
        self.history = self.history[-24:]

    # ---- accuracy (demo mode) -------------------------------------------------------------------
    def error_pct(self, finish_s: float) -> float | None:
        return None if self.truth_s is None else 100 * abs(finish_s - self.truth_s) / self.truth_s

    # ---- the feedback loop ----------------------------------------------------------------------
    def feedback_card(self, turn: dict) -> dict:
        items = [{"race": a["race"], "index": a.get("index"), "old": a["old"], "new": a["new"], "cue": a["cue"],
                  "verdict": a["verdict"],
                  "verdict_text": VERDICT_WORDS.get(a["verdict"], ""), "evidence": a["evidence"]} for a in turn["applied"]]
        st = self.steer.state()
        pr = st["prediction"]
        delta = turn["finish_after_s"] - turn["finish_before_s"]
        card = {"n": turn["n"], "said": turn.get("said", turn["text"]), "items": items, "notes": turn["notes"],
                "target": pr["finish"], "target_delta": None if not items else _signed(delta),
                "range": [format_time(pr["low_s"]), format_time(pr["high_s"])] if pr["low_s"] else None,
                "steering": st["steering"], "suggestions": []}
        if self.truth_s is not None and items:
            before, after = self.error_pct(turn["finish_before_s"]), self.error_pct(turn["finish_after_s"])
            card["accuracy"] = {"closer": after < before - 1e-9, "same": abs(after - before) < 1e-9,
                                "before_pct": before, "after_pct": after,
                                "equal_weights_pct": self.error_pct(pr["initial_s"]),
                                "text": ("closer to her real result" if after < before - 1e-9 else
                                         "further from her real result" if after > before + 1e-9 else
                                         "no closer to her real result")}
        verdicts = {i["verdict"] for i in items}
        if "against_data" in verdicts:
            card["suggestions"] += ["undo", "Why does the data disagree?"]
        if "your_knowledge" in verdicts:
            card["suggestions"] += ["show similar runners", "Explain the likely range"]
        if "supported" in verdicts:
            card["suggestions"] += ["simulate a hot day"]
        if st["steering"]["index"] is not None and st["steering"]["index"] > 0.85:
            card["suggestions"] = ["Is that the data, or me?"] + card["suggestions"]
        if not items and not turn["notes"]:
            card["suggestions"] += ["2016 was a hot day", "trust the recent races more"]
        card["suggestions"] = list(dict.fromkeys(card["suggestions"]))[:4]
        return card

    def feedback_text(self, turn: dict) -> str:
        c = self.feedback_card(turn)
        if not c["items"]:
            return " ".join(c["notes"]) or "Nothing changed."
        parts = [f"{i['race']} {i['old']:.1f} → {i['new']:.1f}: {i['verdict_text']}." for i in c["items"]]
        parts.append(f"Target {c['target']} ({c['target_delta']}).")
        acc = c.get("accuracy")
        if acc and self.revealed:
            parts.append(f"That is {acc['text']} ({acc['after_pct']:.2f}% off).")
        elif acc:
            parts.append(f"That moved the prediction {acc['text'].split(' to ')[0]}.")
        return " ".join(parts)

    def react(self, turn_n: int, value: str, note: str = "") -> dict:
        if value not in ("agree", "unsure", "disagree"):
            raise ValidationError("value must be agree, unsure or disagree")
        self.reactions.append({"turn": int(turn_n), "value": value, "note": note[:500]})
        return self.reaction_summary()

    def reaction_summary(self) -> dict:
        n = len(self.reactions)
        return {"n": n, "agree": sum(r["value"] == "agree" for r in self.reactions),
                "unsure": sum(r["value"] == "unsure" for r in self.reactions),
                "disagree": sum(r["value"] == "disagree" for r in self.reactions)}

    # ---- simulator ------------------------------------------------------------------------------
    def simulate(self, temp_c: float = 12.0, wind_kmh: float = 0.0, profile: str = "on_plan") -> dict:
        pr = self.plan.prediction
        return simulate_race(self.plan.paces, pr["low_s"], pr["high_s"], temp_c, wind_kmh, profile)

    # ---- everything the interface draws ----------------------------------------------------------
    def state(self) -> dict:
        st = self.steer.state()
        pr = self.plan.prediction
        similar = []
        for i, p in enumerate(self._per_race):
            if p is None:
                similar.append(None)
                continue
            vals = np.sort(np.asarray(p.case_finish_s))
            similar.append({"race_index": i, "n": int(p.n_neighbours), "ratio": float(p.mean_ratio),
                            "finishes_s": [float(v) for v in vals],
                            "median_s": float(np.median(vals))})
        st.update({
            "mode": "demo" if self.holdout else "runner",
            "plan": {"paces": list(self.plan.paces), "labels": list(SEGMENT_LABELS), "finish_s": self.plan.plan_finish,
                     "check": {k: self.plan.check.get(k) for k in ("verdict", "note") if k in self.plan.check}},
            "scenarios": [{"key": s["key"], "name": s["name"], "finish": s["finish"]} for s in self.plan.strategies],
            "race_paces": [r.get("paces") for r in self.plan.overview["races"]],
            "similar": similar,
            "insights": pr.get("insights", [])[:4],
            "holdout": None if self.truth_s is None else {
                "label": self.truth_label, "revealed": self.revealed,
                "truth_s": self.truth_s if self.revealed else None,
                "error_pct": self.error_pct(self.plan.model_finish) if self.revealed else None,
                "equal_weights_error_pct": self.error_pct(st["prediction"]["initial_s"]) if self.revealed else None},
            "reactions": self.reaction_summary(),
            "ai": bool(self.ai is not None and getattr(self.ai, "online", False)),
        })
        return st


__all__ = ["HudSession"]
