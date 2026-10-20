"""PaceSense Command Center: a 3D, voice-driven workspace over the same model and services.

    session.py   HudSession: runner, steering, assistant, simulator, feedback loop (immediate + outcome)
    commands.py  one command bar: interface commands, race talk, questions
    sim.py       race-day simulator with stated heat/wind assumptions and the likely-range band
The page is resources/hud.html, served at /hud by `pacesense hud`.
"""
from .commands import route, ui_commands
from .session import HudSession
from .sim import simulate_race

__all__ = ["HudSession", "route", "ui_commands", "simulate_race"]
