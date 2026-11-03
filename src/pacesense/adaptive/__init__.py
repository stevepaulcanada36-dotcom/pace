"""PaceSense Adaptive: N feedback loops, coach persona and a voice-first, self-arranging interface.

See docs/FUTURE_SCOPE.md for how each part answers an open question from Muijlwijk et al. (IUI '24).
"""
from .engine import LAYOUTS, AdaptiveEngine
from .events import Adaptation, Event
from .interview import Question, questions
from .loops import FeedbackLoop, default_loops
from .persona import ARCHETYPES, DIMENSIONS, Persona

__all__ = [
           "ARCHETYPES",
           "DIMENSIONS",
           "LAYOUTS",
           "Adaptation",
           "AdaptiveEngine",
           "Event",
           "FeedbackLoop",
           "Persona",
           "Question",
           "default_loops",
           "questions",
]
