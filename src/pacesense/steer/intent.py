"""Turn what a coach says into race-weight changes, without AI and without randomness.

    interpret("2016 was a hot day and she was injured, trust the recent races more", races)

reads the sentence, finds which race each part is about ("2016", "the last race", "Amsterdam", "it"), finds
what the coach says about it ("hot day", "injured", "ignore", "her best race", "set 2014 to 0.3"), and returns
`Action`s such as "Amsterdam 2016: -0.3 because 'hot day'". The rules live in
`resources/steer_lexicon.yaml`, so they can be read, changed and versioned without touching code.

Design choices (each one testable in tests/test_steer.py):

* Deterministic. The same words and races always give the same actions, so a study can be re-run and a
  coach's session can be replayed from its transcript.
* Every action carries the words it came from (`cue`) and the clause it was in, so the dashboard can show
  *why* a slider moved and the analysis can count changes that came with a reason.
* Nothing is guessed silently: a cue with no race to attach to is returned in `unresolved`, and a race that
  is not in the runner's history (e.g. "2015" when she did not race in 2015) is returned in `unknown_races`.
* Descriptions attach to the race spoken *before* them ("2016 was hot"); commands attach to the race spoken
  *after* them ("ignore 2016"); a sentence with no race ("she was injured") continues the previous race.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable, Literal, Sequence

import yaml

from ..assistant.intake import parse_duration
from ..domain import MAX_FINISH_S, MIN_FINISH_S

Kind = Literal["set", "shift", "reset", "recency", "undo"]

_NUMBER_WORDS = {"zero": 0.0, "nothing": 0.0, "none": 0.0, "half": 0.5, "full": 1.0, "one": 1.0, "max": 1.0,
                 "maximum": 1.0, "minimum": 0.0}
_POINT_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_SET_TRIGGERS = re.compile(r"\b(set|put|make|give|change|move|weight|weigh|slider|to|at)\b")


# ---- the lexicon ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cue:
    phrase: str
    effect: str          # exclude | full | down_strong | down | up | up_strong
    verb: bool           # spoken before the race it refers to


@dataclass(frozen=True)
class Lexicon:
    cues: tuple[Cue, ...]
    effects: dict[str, dict[str, float]]
    soften: tuple[tuple[str, ...], float]
    intensify: tuple[tuple[str, ...], float]
    negations: tuple[str, ...]
    references: dict[str, tuple[str, ...]]
    commands: dict[str, tuple[str, ...]]
    estimate_cues: tuple[str, ...]
    half_life_years: float
    recency_floor: float
    version: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Lexicon":
        cues = []
        for effect, groups in d["phrases"].items():
            for kind in ("verb", "describe"):
                for p in groups.get(kind, []) or []:
                    cues.append(Cue(str(p).lower(), effect, kind == "verb"))
        cues += [Cue("less", "down", False), Cue("more", "up", False)]   # "count 2016 less", "2018 more"
        cues.sort(key=lambda c: -len(c.phrase))
        m = d.get("modifiers", {})
        rec = d.get("recency", {})
        return cls(tuple(cues), d["effects"],
                   (tuple(m["soften"]["words"]), float(m["soften"]["factor"])),
                   (tuple(m["intensify"]["words"]), float(m["intensify"]["factor"])),
                   tuple(d.get("negations", [])),
                   {k: tuple(v) for k, v in d.get("references", {}).items()},
                   {k: tuple(v) for k, v in d.get("commands", {}).items()},
                   tuple(d.get("estimate_cues", [])),
                   float(rec.get("half_life_years", 3.0)), float(rec.get("floor", 0.2)), int(d.get("version", 1)))


def lexicon_text() -> str:
    return resources.files("pacesense.resources").joinpath("steer_lexicon.yaml").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def default_lexicon() -> Lexicon:
    return Lexicon.from_dict(yaml.safe_load(lexicon_text()))


# ---- results ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    kind: Kind
    races: tuple[int, ...]     # indices into the runner's races (empty for reset/undo)
    value: float | None        # target weight for "set", change for "shift"
    cue: str                   # the words that caused it
    clause: str                # the clause it was found in
    confidence: float          # 1.0 explicit number/command, 0.8 a phrase on a named race, 0.6 carried over

    def describe(self, races: Sequence[dict]) -> str:
        names = ", ".join(_race_name(races[i]) for i in self.races) or "all races"
        if self.kind == "set":
            return f"{names} → {self.value:.1f} (\"{self.cue}\")"
        if self.kind == "shift":
            return f"{names} {self.value:+.1f} (\"{self.cue}\")"
        return {"reset": "all races back to 0.5", "undo": "undo the last change",
                "recency": "recent races count more"}[self.kind] + f" (\"{self.cue}\")"


@dataclass
class Interpretation:
    text: str
    actions: list[Action] = field(default_factory=list)
    estimate_s: float | None = None          # the coach's own finish estimate, if stated
    unresolved: list[str] = field(default_factory=list)    # cues with no race to attach to
    unknown_races: list[str] = field(default_factory=list)  # races mentioned that the runner does not have
    context: int | None = None               # the race being talked about at the end (for the next utterance)

    @property
    def understood(self) -> bool:
        return bool(self.actions) or self.estimate_s is not None


# ---- helpers ----------------------------------------------------------------------------------------
def _race_name(r: dict) -> str:
    return f"{r.get('event', 'race')} {r.get('year', '')}".strip()


def _normalise(text: str) -> str:
    t = text.lower().replace("’", "'").replace("–", "-")
    t = re.sub(r"\btwenty (ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\b",
               lambda m: str(2000 + _TEENS[m.group(1)]), t)
    t = re.sub(r"\btwenty twenty(?: (one|two|three|four|five|six|seven|eight|nine))?\b",
               lambda m: str(2020 + (_POINT_WORDS[m.group(1)] if m.group(1) else 0)), t)
    t = re.sub(r"\btwo thousand (?:and )?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
               r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\b",
               lambda m: str(2000 + (_POINT_WORDS.get(m.group(1)) or _TEENS[m.group(1)])), t)
    t = re.sub(r"\bpoint (zero|one|two|three|four|five|six|seven|eight|nine)\b",
               lambda m: f"0.{_POINT_WORDS[m.group(1)]}", t)
    t = re.sub(r"\b'(\d\d)\b", lambda m: f"20{m.group(1)}", t)        # '16 -> 2016
    return re.sub(r"\s+", " ", t).strip()


def _phrase_re(phrase: str) -> re.Pattern:
    return re.compile(r"(?<![\w'])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![\w'])")


@lru_cache(maxsize=4096)
def _compiled(phrase: str) -> re.Pattern:
    return _phrase_re(phrase)


def _find_all(text: str, phrases: Iterable[str], taken: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    """Non-overlapping matches, longest phrase first, skipping spans already taken."""
    out = []
    for p in sorted(set(phrases), key=len, reverse=True):
        for m in _compiled(p).finditer(text):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            out.append((s, e, p))
    return sorted(out)


def _words_before(text: str, pos: int, n: int = 3) -> list[str]:
    return re.findall(r"[\w']+", text[:pos])[-n:]


def _has_phrase_before(text: str, pos: int, phrases: Iterable[str], n_words: int = 3) -> str | None:
    window = " ".join(_words_before(text, pos, n_words))
    for p in sorted(phrases, key=len, reverse=True):
        if _compiled(p).search(window):
            return p
    return None


def _strip_articles(text: str) -> str:
    return re.sub(r"\b(?:the|her|his|their|a|an)\s+", "", text)


def _span_in(text: str, phrase: str) -> tuple[int, int] | None:
    """Where a command phrase (matched without articles) sits in the full text, articles allowed between words."""
    words = phrase.split()
    pat = r"(?<![\w'])" + r"\s+(?:(?:the|her|his|their|a|an)\s+)?".join(re.escape(w) for w in words) + r"(?![\w'])"
    m = re.search(pat, text)
    return m.span() if m else None


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<!\d)[.,](?!\d)|[;!?]+|\s+\bbut\b\s+|\s+\bhowever\b\s+|\s+\bwhereas\b\s+|\s+\balthough\b\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


# ---- race references --------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Ref:
    start: int
    end: int
    races: tuple[int, ...]
    label: str
    pronoun: bool = False


def _order_by_recency(races: Sequence[dict]) -> list[int]:
    """Indices from newest to oldest (by year, then by position: later in the list = more recent)."""
    return sorted(range(len(races)), key=lambda i: (races[i].get("year") or 0, i), reverse=True)


def _references(clause: str, races: Sequence[dict], lex: Lexicon, taken: list[tuple[int, int]],
                unknown: list[str]) -> list[_Ref]:
    refs: list[_Ref] = []
    newest = _order_by_recency(races)
    groups = {
        "most_recent": tuple(newest[:1]),
        "previous": tuple(newest[1:2]),
        "oldest": tuple(newest[-1:]),
        "recent_group": tuple(newest[:2]),
        "older_group": tuple(newest[1:]) if len(newest) <= 3 else tuple(newest[2:]),
        "all": tuple(range(len(races))),
    }
    # event + year ("amsterdam 2016"), then years, then event names, then positional phrases
    events = sorted({str(r.get("event", "")).lower() for r in races if r.get("event")}, key=len, reverse=True)
    for ev in events:
        for m in re.finditer(r"(?<![\w'])" + re.escape(ev) + r"(?:\s+(?:in\s+)?((?:19|20)\d\d))?(?![\w'])", clause):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            year = int(m.group(1)) if m.group(1) else None
            idx = tuple(i for i, r in enumerate(races) if str(r.get("event", "")).lower() == ev
                        and (year is None or r.get("year") == year))
            taken.append((s, e))
            if idx:
                refs.append(_Ref(s, e, idx, m.group(0)))
            else:
                unknown.append(m.group(0))
    for m in re.finditer(r"(?<![\w.])((?:19|20)\d\d)(?![\w.])", clause):
        s, e = m.span()
        if any(s < te and ts < e for ts, te in taken):
            continue
        taken.append((s, e))
        idx = tuple(i for i, r in enumerate(races) if r.get("year") == int(m.group(1)))
        if idx:
            refs.append(_Ref(s, e, idx, m.group(1)))
        else:
            unknown.append(m.group(1))
    for key, phrases in lex.references.items():
        if key == "pronoun":
            continue
        for s, e, p in _find_all(clause, phrases, taken):
            if groups.get(key):
                refs.append(_Ref(s, e, groups[key], p))
    for s, e, p in _find_all(clause, lex.references.get("pronoun", ()), taken):
        refs.append(_Ref(s, e, (), p, pronoun=True))
    return sorted(refs, key=lambda r: r.start)


# ---- explicit numbers ("set 2016 to 0.2") --------------------------------------------------------------
_VALUE = r"(0?\.\d+|1\.0+|0|1|zero|nothing|none|half|full|one|max|maximum|minimum)"


def _explicit_sets(clause: str, refs: list[_Ref], taken: list[tuple[int, int]]) -> list[tuple[_Ref | None, float, str, int]]:
    out = []
    if not _SET_TRIGGERS.search(clause):
        return out
    for m in re.finditer(r"(?:\b(?:to|at|of|=|weight|weighted)\s+(?:a\s+)?(?:weight\s+(?:of\s+)?)?)" + _VALUE + r"(?![\w:])",
                         clause):
        s, e = m.span(1)
        if any(s < te and ts < e for ts, te in taken):
            continue
        raw = m.group(1)
        v = _NUMBER_WORDS.get(raw)
        if v is None:
            try:
                v = float(raw)
            except ValueError:
                continue
        if not 0.0 <= v <= 1.0:
            continue
        taken.append(m.span())
        before = [r for r in refs if r.end <= m.start()]
        target = before[-1] if before else (refs[0] if refs else None)
        out.append((target, round(v, 2), m.group(0).strip(), m.start()))
    return out


# ---- the coach's own estimate ---------------------------------------------------------------------------
_TIME = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?|\d\s*h(?:ours?|rs?)?\s*(?:and\s*)?\d{1,2}\s*(?:m(?:in(?:ute)?s?)?)?(?:\s*\d{1,2}\s*s(?:ec(?:ond)?s?)?)?|"
                   r"(?:sub|under)\s*\d(?::\d{2})?)")


def _estimate(text: str, lex: Lexicon) -> tuple[float | None, tuple[int, int] | None]:
    for cue in lex.estimate_cues:
        for m in _compiled(cue).finditer(text):
            tm = _TIME.search(text, m.end(), m.end() + 40)
            if not tm:
                continue
            raw = tm.group(1)
            raw = re.sub(r"^(?:sub|under)\s*", "", raw)
            raw = re.sub(r"\s*(?:and)\s*", " ", raw)
            if re.fullmatch(r"\d", raw):
                raw += ":00"
            s = parse_duration(raw.replace(" ", ""))
            if s is not None and s < 1000:          # "3:25" read as minutes:seconds -> hours:minutes
                s *= 60
            if s is not None and MIN_FINISH_S <= s <= MAX_FINISH_S:
                return float(s), tm.span()
    return None, None


# ---- main entry point -------------------------------------------------------------------------------------
def interpret(text: str, races: Sequence[dict], context: int | None = None,
              lexicon: Lexicon | None = None) -> Interpretation:
    """Read one utterance. `races` are dicts with at least `event` and `year`, in the order of the sliders.
    `context` is the race index the previous utterance ended on (so "she was injured there" works)."""
    lex = lexicon or default_lexicon()
    norm = _normalise(text)
    out = Interpretation(text=text, context=context)
    if not norm:
        return out

    # whole-utterance commands (matched without articles: "trust the recent races more")
    bare = _strip_articles(norm)
    for kind in ("undo", "reset"):
        hit = _find_all(bare, lex.commands.get(kind, ()), [])
        if hit:
            out.actions.append(Action(kind, (), None, hit[0][2], norm, 1.0))
            return out

    est, est_span = _estimate(norm, lex)
    out.estimate_s = est
    rec = _find_all(bare, lex.commands.get("recency", ()), [])
    recency_spans = [_span_in(norm, p) for _, _, p in rec]
    recency_spans = [x for x in recency_spans if x]
    if rec:
        out.actions.append(Action("recency", tuple(range(len(races))), None, rec[0][2], norm, 1.0))

    ctx = context
    for clause in _sentences(norm):
        offset = norm.find(clause)
        taken: list[tuple[int, int]] = []
        for s, e in recency_spans + ([est_span] if est_span else []):     # already used: do not read again
            if s >= offset and e <= offset + len(clause) + 1:
                taken.append((s - offset, e - offset))
        unknown: list[str] = []
        refs = _references(clause, races, lex, taken, unknown)
        out.unknown_races += unknown
        # pronouns point at the latest named race before them, else the conversation's context
        resolved: list[_Ref] = []
        for r in refs:
            if r.pronoun:
                prior = [x for x in resolved if not x.pronoun]
                idx = prior[-1].races if prior else ((ctx,) if ctx is not None else ())
                if idx:
                    resolved.append(_Ref(r.start, r.end, idx, r.label, True))
            else:
                resolved.append(r)
        named = [r for r in resolved if r.races]

        for target, value, cue, _pos in _explicit_sets(clause, named, taken):
            races_idx = target.races if target else ((ctx,) if ctx is not None else ())
            if races_idx:
                out.actions.append(Action("set", races_idx, value, cue, clause, 1.0))
                ctx = races_idx[-1]
            else:
                out.unresolved.append(cue)

        cue_hits = _find_all(clause, [c.phrase for c in lex.cues], taken)
        by_phrase = {c.phrase: c for c in lex.cues}
        for s, e, phrase in cue_hits:
            cue = by_phrase[phrase]
            spec = lex.effects[cue.effect]
            neg = _has_phrase_before(clause, s, lex.negations, 3)
            mod = 1.0
            if _has_phrase_before(clause, s, lex.soften[0], 3):
                mod = lex.soften[1]
            elif _has_phrase_before(clause, s, lex.intensify[0], 2):
                mod = lex.intensify[1]
            before = [r for r in named if r.end <= s]
            after = [r for r in named if r.start >= e]
            if cue.verb:
                target = after[0] if after else (before[-1] if before else None)
            else:
                target = before[-1] if before else (after[0] if after else None)
            conf = 0.8
            if target is None:
                if ctx is None:
                    out.unresolved.append(phrase)
                    continue
                races_idx, conf = (ctx,), 0.6
            else:
                races_idx = target.races
                if target.pronoun:
                    conf = 0.6
            label = (f"{neg} {phrase}" if neg else phrase)
            if "set" in spec and not neg:
                out.actions.append(Action("set", races_idx, float(spec["set"]), label, clause, conf))
            else:
                delta = float(spec.get("shift", -0.3 if spec.get("set", 1) == 0 else 0.3))
                if neg:
                    delta = -0.5 * delta
                out.actions.append(Action("shift", races_idx, round(delta * mod, 2), label, clause, conf))
            ctx = races_idx[-1] if len(races_idx) == 1 else ctx
        if named and not cue_hits:
            last = named[-1]
            ctx = last.races[-1] if len(last.races) == 1 else ctx
    out.context = ctx
    out.actions = _merge(out.actions)
    return out


def _merge(actions: list[Action]) -> list[Action]:
    """One action per race per utterance: shifts add up ("hot and injured" = -0.6); a set wins over shifts."""
    singles: dict[int, list[Action]] = {}
    keep: list[Action] = []
    for a in actions:
        if a.kind in ("set", "shift"):
            for i in a.races:
                singles.setdefault(i, []).append(a)
        else:
            keep.append(a)
    for i, acts in singles.items():
        sets = [a for a in acts if a.kind == "set"]
        if sets:
            a = sets[-1]
            keep.append(Action("set", (i,), a.value, a.cue, a.clause, a.confidence))
        else:
            total = max(-1.0, min(1.0, sum(a.value or 0 for a in acts)))
            cues = " + ".join(dict.fromkeys(a.cue for a in acts))
            keep.append(Action("shift", (i,), round(total, 2), cues, acts[-1].clause, min(a.confidence for a in acts)))
    order = {"undo": 0, "reset": 1, "recency": 2, "set": 3, "shift": 3}
    return sorted(keep, key=lambda a: (order[a.kind], a.races))


def recency_weights(years: Sequence[int | None], half_life: float, floor: float = 0.2) -> list[float]:
    """2 ** (-years before the latest race / half_life), clipped to [floor, 1], rounded to the slider step."""
    ys = [y for y in years if y is not None]
    latest = max(ys) if ys else None
    out = []
    for y in years:
        if latest is None or y is None or half_life == float("inf"):
            out.append(0.5 if half_life == float("inf") else 1.0)
            continue
        w = 2 ** (-(latest - y) / half_life)
        out.append(round(min(1.0, max(floor, w)), 1))
    return out


__all__ = ["Action", "Interpretation", "Lexicon", "default_lexicon", "interpret", "lexicon_text", "recency_weights"]
