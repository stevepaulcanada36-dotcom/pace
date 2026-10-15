"""Ask PaceSense: an AI assistant that analyses runners and plans with PaceSense's own functions.

    intake.py   check and correct runner details (no AI)
    tools.py    the PaceSense functions the AI may call, and their descriptions
    chat.py     the conversation loop and the OpenRouter chat model
    offline.py  a no-AI helper with the same shape, for offline use and tests
"""
from .chat import Answer, Assistant, AssistantError, OpenRouterModel, Step
from .intake import check_runner, parse_duration
from .offline import OfflineModel
from .tools import AssistantTools

__all__ = ["Answer", "Assistant", "AssistantError", "AssistantTools", "OfflineModel", "OpenRouterModel", "Step",
           "check_runner", "parse_duration"]
