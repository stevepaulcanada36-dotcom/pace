"""The study procedure (Muijlwijk et al. 2024, Section 4.1) as a service.

Per coach: a session in the interactive or non-interactive condition with four
runners -- two familiar (the coach's own) and two unfamiliar, in random block
order, or four unfamiliar if the coach continues without their own runners.
Per runner: Coach_pre -> Model_initial -> (Model_adapted) -> Coach_post.
For unfamiliar runners the best of the five most recent races is hidden and
kept as ground truth.
"""
from __future__ import annotations

import logging
import time
import uuid
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from .cbr import DEFAULT_WEIGHT, MarathonModel, Prediction, holdout_best
from .domain import (SEGMENT_KM, Race, Runner, ValidationError, format_time, is_valid_finish,
                     linear_pacing_profile, pace_per_km)
from .measures import compute_trial_measures
from .questionnaire import Questionnaire
from .storage import Repository

log = logging.getLogger(__name__)

RUNNERS_PER_SESSION = 4
Assignment = Literal["random", "interactive", "noninteractive"]

RUNNER_MOTIVATIONS = ("General health", "Weight concern", "Affiliation", "Recognition", "Psychological coping",
                      "Life meaning", "Self-esteem", "Competition", "Personal goal achievement")


_PSEUDONYMS = {
    "F": ("Ann", "Eva", "Lisa", "Sanne", "Iris", "Marieke", "Femke", "Lotte", "Noor", "Inge", "Anouk", "Ellen"),
    "M": ("Bas", "Tom", "Joost", "Pieter", "Daan", "Erik", "Mark", "Sander", "Koen", "Ruben", "Lars", "Jeroen"),
}


def pseudonym(runner_id: str, gender: str) -> str:
    names = _PSEUDONYMS["F" if gender == "F" else "M"]
    return names[zlib.crc32(runner_id.encode()) % len(names)]


class NotFound(LookupError):
    pass


class Conflict(RuntimeError):
    pass


# serialisation helpers ---------------------------------------------------------
def race_to_dict(r: Race) -> dict:
    return {"event": r.event, "year": r.year, "finish_s": r.finish_s, "age": r.age,
            "segments_s": list(r.segments_s) if r.segments_s else None}


def race_from_dict(d: dict) -> Race:
    return Race(d["event"], int(d["year"]), float(d["finish_s"]), d.get("age"),
                tuple(d["segments_s"]) if d.get("segments_s") else None)


def runner_snapshot(runner: Runner, shown: Sequence[Race]) -> dict:
    return {"runner_id": runner.runner_id, "name": runner.name, "gender": runner.gender, "age": runner.age,
            "races": [race_to_dict(r) for r in shown]}


def runner_from_snapshot(d: dict) -> tuple[Runner, list[Race]]:
    races = [race_from_dict(x) for x in d["races"]]
    return Runner(d["runner_id"], d["gender"], list(races), name=d.get("name") or "", age=d.get("age")), races


def prediction_view(pred: Prediction) -> dict:
    paces = pace_per_km(pred.finish_s, pred.profile)
    return {
        "finish_s": round(pred.finish_s, 1),
        "finish": format_time(pred.finish_s),
        "weights": list(pred.weights),
        "pace_per_km_s": [round(float(p), 1) for p in paces],
        "profile": [round(float(x), 5) for x in pred.profile],
        "per_race": [None if p is None else {"finish_s": round(p.finish_s, 1), "neighbours": p.n_neighbours,
                                             "profile": None if p.profile is None else [round(float(x), 5) for x in p.profile]}
                     for p in pred.per_race],
    }


@dataclass
class StudyService:
    model: MarathonModel
    runners: Sequence[Runner]
    repo: Repository
    data_source: str = "unknown"
    seed: int | None = None
    questionnaire: Questionnaire | None = None

    def __post_init__(self) -> None:
        if self.questionnaire is None:
            self.questionnaire = Questionnaire.load()
        self._rng = np.random.default_rng(self.seed)
        self._by_id = {r.runner_id: r for r in self.runners}
        self._unfamiliar_pool = [r for r in self.runners if len(r.races) >= 3]
        if not self._unfamiliar_pool:
            raise ValidationError("the race database has no runners with 3 or more races")

    # lookup -----------------------------------------------------------------------
    def search_runners(self, query: str, limit: int = 10) -> list[dict]:
        q = query.strip().lower()
        if len(q) < 2:
            return []
        hits = []
        for r in self.runners:
            if q in r.runner_id.lower() or (r.name and q in r.name.lower()):
                hits.append({"runner_id": r.runner_id, "name": r.name, "gender": r.gender, "age": r.age,
                             "races": [{"event": x.event, "year": x.year, "finish": format_time(x.finish_s)}
                                       for x in r.recent(5)]})
                if len(hits) >= limit:
                    break
        return hits

    def runner(self, runner_id: str) -> Runner:
        try:
            return self._by_id[runner_id]
        except KeyError:
            raise NotFound(f"runner {runner_id!r} not found") from None

    # session lifecycle -------------------------------------------------------------
    def create_session(self, assignment: Assignment = "random", coach_label: str | None = None) -> dict:
        if assignment not in ("random", "interactive", "noninteractive"):
            raise ValidationError("assignment must be random, interactive or noninteractive")
        condition = assignment if assignment != "random" else ("interactive" if self._rng.random() < .5 else "noninteractive")
        sid = uuid.uuid4().hex
        self.repo.insert_session(id=sid, condition=condition, coach_label=coach_label, data_source=self.data_source,
                                 created_at=time.time())
        log.info("session %s created (%s)", sid, condition)
        return {"session_id": sid, "condition": condition}

    def _session(self, sid: str) -> dict:
        s = self.repo.get_session(sid)
        if s is None:
            raise NotFound(f"session {sid!r} not found")
        return s

    def plan_runners(self, sid: str, familiar: Iterable[Runner | str] = (), unfamiliar_only: bool = False) -> list[dict]:
        self._session(sid)
        if self.repo.list_trials(sid):
            raise Conflict("runners have already been chosen for this session")
        fam = [self.runner(f) if isinstance(f, str) else f for f in familiar]
        if unfamiliar_only:
            fam = []
        elif len(fam) != 2:
            raise ValidationError("choose exactly two familiar runners, or continue with unfamiliar runners only")
        if len({r.runner_id for r in fam}) != len(fam):
            raise ValidationError("the two familiar runners must be different")
        for r in fam:
            if not r.races:
                raise ValidationError(f"runner {r.runner_id} has no races")
        unf = self._draw_unfamiliar(RUNNERS_PER_SESSION - len(fam), sid, exclude={r.runner_id for r in fam},
                                    match=[len(r.recent(5)) for r in fam])
        rows: list[dict[str, Any]] = []
        fam_rows = [self._trial_row(sid, r, familiar=True) for r in fam]
        unf_rows = [self._trial_row(sid, r, familiar=False) for r in unf]
        ordered = fam_rows + unf_rows if self._rng.random() < .5 else unf_rows + fam_rows
        for i, row in enumerate(ordered, start=1):
            row["seq"] = i
            rows.append(row)
        self.repo.insert_trials(rows)
        self.repo.update_session(sid, unfamiliar_only=int(not fam))
        return [self.trial_view(sid, r["seq"]) for r in rows]

    def _draw_unfamiliar(self, n: int, sid: str, exclude: set[str], match: list[int]) -> list[Runner]:
        """Sec. 4.1: reuse runners that previous participants selected as their own, choosing the most
        recently selected ones with the same number of shown marathons as this coach's own runners.
        Before any such runners exist, draw from the prepared list (database runners with 3+ races)."""
        chosen: list[Runner] = []
        taken = set(exclude)
        previous = []
        for snap in self.repo.previous_familiar_runners(exclude_session=sid):
            runner, races = runner_from_snapshot(snap)
            if len(races) >= 3 and runner.runner_id not in taken:
                previous.append(runner)
        for i in range(n):
            n_shown = match[i] if i < len(match) else None
            def fits(r: Runner) -> bool:
                return r.runner_id not in taken and (n_shown is None or len(r.recent(5)) - 1 == n_shown)
            pick = next((r for r in previous if fits(r)), None)
            if pick is None:
                pool = [r for r in self._unfamiliar_pool if fits(r)] or \
                       [r for r in self._unfamiliar_pool if r.runner_id not in taken]
                pick = pool[int(self._rng.integers(len(pool)))]
            chosen.append(pick)
            taken.add(pick.runner_id)
        return chosen

    @staticmethod
    def _trial_row(sid: str, runner: Runner, familiar: bool) -> dict[str, Any]:
        if familiar:
            shown, truth = runner.recent(5), None
        else:
            best, shown = holdout_best(runner)
            truth = best.finish_s
        return {"session_id": sid, "status": "planned", "familiar": int(familiar), "runner_id": runner.runner_id,
                "runner": runner_snapshot(runner, shown), "truth_s": truth}

    def _trial(self, sid: str, seq: int) -> dict:
        t = self.repo.get_trial(sid, seq)
        if t is None:
            raise NotFound(f"runner {seq} not found in session {sid}")
        return t

    def _require_turn(self, sid: str, seq: int) -> None:
        for t in self.repo.list_trials(sid):
            if t["seq"] < seq and t["status"] != "done":
                raise Conflict(f"finish runner {t['seq']} first")

    def trial_view(self, sid: str, seq: int) -> dict:
        s, t = self._session(sid), self._trial(sid, seq)
        snap = t["runner"]
        view = {"seq": seq, "of": RUNNERS_PER_SESSION, "status": t["status"], "condition": s["condition"],
                "familiar": bool(t["familiar"]), "gender": snap["gender"], "age": snap["age"],
                "name": (snap["name"] or snap["runner_id"]) if t["familiar"] else pseudonym(snap["runner_id"], snap["gender"]),
                "races": [{"event": r["event"], "year": r["year"], "finish_s": r["finish_s"],
                           "finish": format_time(r["finish_s"]),
                           "pace_per_km_s": None if not r["segments_s"] else
                           [round(seg / km, 1) for seg, km in zip(r["segments_s"], SEGMENT_KM)]}
                          for r in snap["races"]]}
        if t["status"] != "planned":
            view["coach_pre_s"] = t["coach_pre"]
            view["model"] = prediction_view(self._predict(t, t["weights"]))
            view["model_initial_s"] = t["model_initial"]
        return view

    def _predict(self, trial: dict, weights: Sequence[float] | None) -> Prediction:
        runner, shown = runner_from_snapshot(trial["runner"])
        return self.model.predict(runner, shown, weights)

    def submit_pre(self, sid: str, seq: int, coach_pre_s: float, pace_pre: float = 0.0,
                   familiarity: int | None = None, runner_motivations: Sequence[str] = ()) -> dict:
        t = self._trial(sid, seq)
        self._require_turn(sid, seq)
        if t["status"] != "planned":
            raise Conflict("your first estimate for this runner was already recorded")
        if not is_valid_finish(coach_pre_s):
            raise ValidationError("finish time must be between 2:00:00 and 7:00:00")
        linear_pacing_profile(pace_pre)
        if t["familiar"]:
            if familiarity not in (1, 2, 3, 4, 5):
                raise ValidationError("familiarity must be 1-5 for your own runners")
            bad = set(runner_motivations) - set(RUNNER_MOTIVATIONS)
            if bad:
                raise ValidationError(f"unknown runner motivations: {sorted(bad)}")
        pred = self._predict(t, None)
        self.repo.update_trial(sid, seq, status="pre", coach_pre=float(coach_pre_s), pace_pre=float(pace_pre),
                               familiarity=familiarity if t["familiar"] else None,
                               runner_motivations=list(runner_motivations) if t["familiar"] else None,
                               model_initial=pred.finish_s, model_adapted=pred.finish_s,
                               weights=[DEFAULT_WEIGHT] * len(pred.weights), started_at=time.time())
        return prediction_view(pred)

    def set_weights(self, sid: str, seq: int, weights: Sequence[float]) -> dict:
        s, t = self._session(sid), self._trial(sid, seq)
        if s["condition"] != "interactive":
            raise Conflict("the model can't be adjusted in this condition")
        if t["status"] != "pre":
            raise Conflict("the model can only be adjusted after your first estimate and before your final advice")
        rounded = [round(float(w), 1) for w in weights]
        pred = self._predict(t, rounded)
        self.repo.update_trial(sid, seq, weights=rounded, model_adapted=pred.finish_s)
        self.repo.increment_interactions(sid, seq)
        return prediction_view(pred)

    def submit_post(self, sid: str, seq: int, coach_post_s: float, pace_post: float = 0.0, motivation: str = "") -> dict:
        s, t = self._session(sid), self._trial(sid, seq)
        if t["status"] != "pre":
            raise Conflict("give your first estimate before your final advice" if t["status"] == "planned"
                           else "your final advice for this runner was already recorded")
        if not is_valid_finish(coach_post_s):
            raise ValidationError("finish time must be between 2:00:00 and 7:00:00")
        linear_pacing_profile(pace_post)
        interactive = s["condition"] == "interactive"
        m = compute_trial_measures(interactive=interactive, coach_pre_s=t["coach_pre"], model_initial_s=t["model_initial"],
                                   model_adapted_s=t["model_adapted"], coach_post_s=float(coach_post_s), truth_s=t["truth_s"])
        self.repo.update_trial(sid, seq, status="done", coach_post=float(coach_post_s), pace_post=float(pace_post),
                               motivation=(motivation or "")[:5000], measures=m.as_dict(), ended_at=time.time())
        return {"measures": m.as_dict(), "next": seq + 1 if seq < RUNNERS_PER_SESSION else "questionnaire"}

    def submit_questionnaire(self, sid: str, answers: dict[str, Any]) -> dict:
        s = self._session(sid)
        trials = self.repo.list_trials(sid)
        if len(trials) != RUNNERS_PER_SESSION or any(t["status"] != "done" for t in trials):
            raise Conflict("finish all four runners before the questionnaire")
        if s["completed_at"]:
            raise Conflict("the questionnaire was already submitted")
        clean = self.questionnaire.validate(answers, s["condition"])
        self.repo.update_session(sid, questionnaire=clean, completed_at=time.time())
        return {"completed": True, "perceived_competence": clean.get("perceived_competence"),
                "willingness_to_depend": clean.get("willingness_to_depend")}

    def questionnaire_for(self, sid: str) -> dict:
        return self.questionnaire.for_condition(self._session(sid)["condition"])

    def session_summary(self, sid: str) -> dict:
        s = self._session(sid)
        return {"session_id": sid, "condition": s["condition"], "completed": bool(s["completed_at"]),
                "runners": [{"seq": t["seq"], "familiar": bool(t["familiar"]), "status": t["status"],
                             "coach_pre": format_time(t["coach_pre"]), "model_initial": format_time(t["model_initial"]),
                             "model_adapted": format_time(t["model_adapted"]) if s["condition"] == "interactive" else None,
                             "coach_post": format_time(t["coach_post"]),
                             "best_recent_race": format_time(t["truth_s"]) if t["truth_s"] and t["status"] == "done" else None}
                            for t in self.repo.list_trials(sid)]}
