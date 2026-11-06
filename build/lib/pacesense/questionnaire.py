"""Post-study questionnaire (Sec. 4.2.1 trust, Sec. 4.2.3 covariates), loaded from JSON."""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .domain import ValidationError


@dataclass(frozen=True)
class Questionnaire:
    spec: dict

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Questionnaire":
        if path is None:
            text = resources.files("pacesense.resources").joinpath("questionnaire.json").read_text("utf-8")
        else:
            text = Path(path).read_text("utf-8")
        spec = json.loads(text)
        ids = [i["id"] for s in spec["scales"] for i in s["items"]] + [d["id"] for d in spec["demographics"]]
        if len(ids) != len(set(ids)):
            raise ValidationError("questionnaire item ids must be unique")
        return cls(spec)

    def scales_for(self, condition: str) -> list[dict]:
        return [s for s in self.spec["scales"] if s.get("condition") in (None, condition)]

    def for_condition(self, condition: str) -> dict:
        return {"scales": self.scales_for(condition), "demographics": self.spec["demographics"]}

    @property
    def item_scores(self) -> dict[str, str]:
        return {i["id"]: i["score"] for s in self.spec["scales"] for i in s["items"]}

    def validate(self, answers: dict[str, Any], condition: str) -> dict[str, Any]:
        """Check answers and add one score per construct (mean of its items)."""
        clean: dict[str, Any] = {}
        for scale in self.scales_for(condition):
            for item in scale["items"]:
                v = answers.get(item["id"])
                if not isinstance(v, (int, float)) or isinstance(v, bool) or int(v) != v or not 1 <= v <= scale["points"]:
                    raise ValidationError(f"answer '{item['id']}' must be a whole number from 1 to {scale['points']}")
                clean[item["id"]] = int(v)
        for d in self.spec["demographics"]:
            v = answers.get(d["id"])
            if v is None or v == "":
                continue
            if d["type"] == "number":
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not d["min"] <= v <= d["max"]:
                    raise ValidationError(f"'{d['id']}' must be a number from {d['min']} to {d['max']}")
            elif v not in d["options"]:
                raise ValidationError(f"'{d['id']}' must be one of {d['options']}")
            clean[d["id"]] = v
        if answers.get("comment"):
            clean["comment"] = str(answers["comment"])[:5000]
        groups: dict[str, list[int]] = {}
        for item_id, score in self.item_scores.items():
            if item_id in clean:
                groups.setdefault(score, []).append(clean[item_id])
        for score, vals in groups.items():
            clean[score] = float(np.mean(vals))
        return clean
