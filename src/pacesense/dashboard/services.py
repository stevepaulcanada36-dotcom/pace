"""Builds the model and services once per server process, from environment settings.

Environment:
    PACESENSE_DATA          race CSV (default: synthetic demo database)
    PACESENSE_DB            SQLite file for plans, runners, settings and the study (default: pacesense.sqlite3)
    PACESENSE_ADMIN_TOKEN   unlocks the Study results page
    PACESENSE_ASSIGNMENT    study condition: random (default), interactive or noninteractive
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..tuning import configured_cbr
from ..cbr import CaseBase, MarathonModel
from ..coach import CoachService
from ..data import load_csv, synthetic_runners
from ..storage import Repository
from ..study import StudyService


@dataclass(frozen=True)
class Services:
    coach: CoachService
    study: StudyService
    repo: Repository
    data_source: str
    admin_token: str | None
    assignment: str


def build_services(db_path: str = "pacesense.sqlite3", data_path: str | None = None, *, seed: int | None = None,
                   admin_token: str | None = None, assignment: str = "random") -> Services:
    """Load the race data, build the case base and open the database. Plain Python: no Streamlit needed."""
    if data_path:
        runners, _ = load_csv(data_path)
        source = os.path.basename(data_path)
    else:
        runners, source = synthetic_runners(), "synthetic demo data"
    model = MarathonModel(CaseBase(runners, configured_cbr()))
    repo = Repository(db_path)
    study = StudyService(model, runners, repo, data_source=source, seed=seed)
    return Services(CoachService(model, runners, repo), study, repo, source, admin_token or None, assignment)


def services_from_env() -> Services:
    return build_services(os.environ.get("PACESENSE_DB", "pacesense.sqlite3"), os.environ.get("PACESENSE_DATA") or None,
                          admin_token=os.environ.get("PACESENSE_ADMIN_TOKEN"),
                          assignment=os.environ.get("PACESENSE_ASSIGNMENT", "random"))
