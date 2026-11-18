"""Answering PaceSense's questions by voice.

When a question is open ("Which race was hot?", "Shall I count it less?"), what the coach says next is first
matched against the choices on screen, so the conversation never needs a tap:

    "yes" / "sure" / "go ahead"           -> the choice marked yes        (Tandem's spoken-confirmation rule:
    "no" / "leave it" / "skip"            -> the choice marked no          a yes only counts if it was said)
    "the second one" / "number 2"         -> by position
    "2010" / "Amsterdam" / "undo"         -> by the words of a choice's label
    anything longer that matches nothing  -> not an answer; it is handled as a new sentence

Matching is deterministic and conservative: ambiguous answers (two choices fit equally) return None.
"""
from __future__ import annotations

import re

YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|go ahead|do it|please do|absolutely|correct|that's right|"
                 r"sounds good|right|exactly|definitely|of course|yes please)\b")
NO = re.compile(r"^(no|nope|nah|don't|do not|leave it|keep it as it is|skip|never mind|not really|cancel|"
                r"no thanks|not now|neither|none|not sure|no idea|i don't know|dunno)\b")
ORDINALS = {"first": 0, "1st": 0, "one": 0, "second": 1, "2nd": 1, "two": 1, "third": 2, "3rd": 2, "three": 2,
            "fourth": 3, "4th": 3, "four": 3, "fifth": 4, "5th": 4, "five": 4}
ORDINAL = re.compile(r"\b(?:the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last)\b(?:\s+one)?|"
                     r"\b(?:number|option|choice)\s+(one|two|three|four|five|\d)\b")
STOP = {"the", "a", "an", "it", "is", "was", "that", "this", "of", "to", "and", "i", "me", "my", "her", "his",
        "she", "he", "show", "what", "yes", "no", "s", "one", "just", "please", "ok"}


def _norm(text: str) -> str:
    return re.sub(r"[^\w' ]+", " ", text.lower()).strip()


def _words(text: str) -> set[str]:
    return {w for w in _norm(text).split() if w not in STOP and len(w) > 1}


def is_bare_answer(text: str) -> bool:
    """Just "yes", "no thanks", "the first one"…: an answer with no question to answer."""
    t = _norm(text)
    return len(t.split()) <= 3 and bool(YES.match(t) or NO.match(t) or ORDINAL.search(t))


def match_choice(text: str, choices: list[dict]) -> dict | None:
    """The choice the coach picked by saying `text`, or None if it isn't an answer to this question."""
    if not choices or not text or not text.strip():
        return None
    t = _norm(text)
    n_words = len(t.split())
    if YES.match(t) and n_words <= 6:
        yes = [c for c in choices if c.get("yes")]
        if len(yes) == 1:
            return yes[0]
    if NO.match(t) and n_words <= 6:
        no = [c for c in choices if c.get("no")] or [c for c in choices if c.get("action") == "dismiss"]
        if len(no) == 1:
            return no[0]
    m = ORDINAL.search(t)
    if m and n_words <= 6:
        key = m.group(1) or m.group(2)
        i = len(choices) - 1 if key == "last" else int(key) - 1 if key.isdigit() else ORDINALS.get(key)
        if i is not None and 0 <= i < len(choices):
            return choices[i]
    said = _words(t)
    if not said:
        return None
    scored = []
    for c in choices:
        label = _words(c.get("label", ""))
        if not label:
            continue
        hit = said & label
        # A whole sentence that merely shares a word with a choice ("the data is not accurate" and
        # "Not at all") is not an answer to the question: it must contain the choice, or a year.
        if n_words > len(label) + 3 and hit != label and not any(w.isdigit() and len(w) == 4 for w in hit):
            continue
        # a year is decisive on its own; the first word of a label ("keep", "undo", "show") counts extra;
        # otherwise most of the label must be said
        decisive = any(w.isdigit() and len(w) == 4 for w in hit)
        first = next((w for w in _norm(c.get("label", "")).split() if w in label), None)
        score = len(hit) / len(label) + (1 if decisive else 0) + (0.5 if first in hit else 0)
        if hit and (decisive or score >= 0.5):
            scored.append((score, c))
    if not scored or n_words > 12:
        return None
    scored.sort(key=lambda x: -x[0])
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-9:
        return None                                       # ambiguous: "Amsterdam" when there are two
    return scored[0][1]


__all__ = ["is_bare_answer", "match_choice"]
