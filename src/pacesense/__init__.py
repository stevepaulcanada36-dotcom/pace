"""PaceSense: interactive case-based marathon prediction for running coaches.

Implements the system and study design of Muijlwijk, Willemsen, Smyth and
IJsselsteijn (2024), "Benefits of Human-AI Interaction for Expert Users
Interacting with Prediction Models: a Study on Marathon Running", IUI '24.
"""
from .domain import Race, Runner, format_time, parse_time
from .cbr import CBRConfig, CaseBase, MarathonModel, Prediction, holdout_best, validate_model
from .measures import acceptance, adaptation_class, model_adaptation, percent_error

__all__ = [
    "Race", "Runner", "format_time", "parse_time",
    "CBRConfig", "CaseBase", "MarathonModel", "Prediction", "holdout_best", "validate_model",
    "acceptance", "adaptation_class", "model_adaptation", "percent_error",
]
__version__ = "3.0.0"
