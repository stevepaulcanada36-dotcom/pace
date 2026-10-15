"""PaceSense Live: steer the model's race weights by talking, with the data checking every change.

    intent.py    words -> actions (rules in resources/steer_lexicon.yaml; deterministic)
    evidence.py  race evidence, evidence check per change, influence, steering index
    recency.py   fit the "trust recent races more" profile with the paper's hold-out protocol
    session.py   SteeringSession: applies utterances to a PlanSession, keeps reasons, undo, replay
    benchmark.py scores the interpreter on resources/steer_benchmark.yaml
"""
from .evidence import check_change, influence, race_evidence, steering_index
from .intent import Action, Interpretation, default_lexicon, interpret, recency_weights
from .session import SteeringSession, Turn, replay

__all__ = ["Action", "Interpretation", "SteeringSession", "Turn", "check_change", "default_lexicon", "influence",
           "interpret", "race_evidence", "recency_weights", "replay", "steering_index"]
